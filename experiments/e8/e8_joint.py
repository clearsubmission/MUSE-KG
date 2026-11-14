"""E8: joint listener-budget + creator-aware controller (E5 mixed-relevance split).

Score for a new (cold) artist i shown to user u under genre request t:
    s'_ui = s_ui + gamma + eta * g(m_i)
  s_ui  : cosine score of the anchor-placed embedding (as in E6, beta=0 placement)
  gamma : uniform discovery bonus (listener-side; gamma-only = the E6/E7 budget controller)
  g(m_i): creator-side bonus from the artist's tag count m_i, known at arrival:
          1.0 / 0.5 / 0.0 for the lowest / middle / highest tag-count tercile of new artists
  cap   : optional cap on each new artist's tags before placement (0 = no cap)
Ranking reproduces recommend_constrained() of E6 exactly (>=16 of 20 slots on the requested tag),
with randomized tie-breaking. Usage:
  python e8_joint.py --split validation --controls rock,jazz --out results/e8
"""
import argparse, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

p = argparse.ArgumentParser()
p.add_argument("--split", choices=["validation", "test"], required=True)
p.add_argument("--controls", default="all")
p.add_argument("--out", default="results/e8")
p.add_argument("--ckpt", default="results/e5_mixed_relevance/e5_bpr_checkpoint.pt")
p.add_argument("--gammas", default="0,0.005,0.01,0.02,0.03,0.05,0.075,0.1")
p.add_argument("--etas", default="0,0.01,0.02,0.03,0.05,0.075,0.1,0.15")
p.add_argument("--caps", default="0,4")
p.add_argument("--tie_seed", type=int, default=0)
p.add_argument("--cap_seed", type=int, default=1)
a = p.parse_args()

ROOT = Path(os.environ.get("MUSE_ROOT", "/storage/bhull113/MUSE-KG"))
DATA = ROOT / "processed/e5_mixed_relevance"
OUT = ROOT / a.out; OUT.mkdir(parents=True, exist_ok=True)
K, ANCHORS, TARGET = 20, 20, 16
CONTROLS = {"rock": 73, "electronic": 18, "pop": 24, "indie": 81, "metal": 1, "jazz": 83, "ambient": 14, "acoustic": 185}
if a.controls != "all":
    CONTROLS = {c: CONTROLS[c] for c in a.controls.split(",")}
GAMMAS = [float(x) for x in a.gammas.split(",")]
ETAS = [float(x) for x in a.etas.split(",")]
CAPS = [int(x) for x in a.caps.split(",")]

ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
U = ck["model_state"]["user_emb.weight"].numpy(); V = ck["model_state"]["item_emb.weight"].numpy()
users = np.asarray(ck["users"], dtype=int); items = np.asarray(ck["items"], dtype=int)
umap = {int(u): j for j, u in enumerate(users)}
train = pd.read_csv(DATA / "train.csv")
heldout = pd.read_csv(DATA / f"{a.split}.csv")
cold_df = pd.read_csv(DATA / f"cold_{a.split}.csv"); warm_df = pd.read_csv(DATA / f"warm_{a.split}.csv")
assert len(cold_df) + len(warm_df) == len(heldout)
tags_all = pd.read_csv(DATA / "tags_without_direct_heldout_pairs.csv")[["item_id", "tag_id"]].drop_duplicates()
cold = np.sort(cold_df.item_id.unique())
cand = np.concatenate([items, cold]); nI, nC = len(items), len(cold)
assert len(np.unique(cand)) == len(cand)
cmap = {int(i): j for j, i in enumerate(cand)}
all_tags = np.sort(tags_all.tag_id.unique()); tmap = {int(t): j for j, t in enumerate(all_tags)}
tags_all = tags_all[tags_all.item_id.isin(cmap)].copy()

def tag_matrix(tg):
    A = csr_matrix((np.ones(len(tg), np.float32), (tg.item_id.map(cmap).to_numpy(int), tg.tag_id.map(tmap).to_numpy(int))),
                   shape=(len(cand), len(all_tags)))
    A.data[:] = 1.0
    return A

A_full = tag_matrix(tags_all)               # used for genre indicators and tag counts
cold_set = set(map(int, cold))
m_cold = np.asarray((A_full[nI:] > 0).sum(axis=1)).ravel()  # tag count per new artist
# creator signal: tag-count terciles among the new artists (ties by item id), fewest tags -> bonus 1
order = np.lexsort((cold, m_cold)); terc = np.empty(nC, int)
terc[order] = np.minimum(np.arange(nC) * 3 // nC, 2)
g = np.array([1.0, 0.5, 0.0])[terc].astype(np.float32)

def cold_vectors(cap):
    tg = tags_all
    if cap > 0:
        is_cold = tg.item_id.isin(cold_set)
        c = tg[is_cold].sample(frac=1, random_state=a.cap_seed)
        c = c[c.groupby("item_id").cumcount() < cap]
        tg = pd.concat([tg[~is_cold], c], ignore_index=True)
    A = tag_matrix(tg)
    df = np.asarray((A[:nI] > 0).sum(axis=0)).ravel()
    idf = np.log((nI + 1) / (df + 1)) + 1.0                      # IDF from training artists only
    At = normalize(A.multiply(idf).tocsr(), axis=1)
    sims = (At[nI:] @ At[:nI].T).tocsr()
    out = np.zeros((nC, V.shape[1]), np.float32)
    for r in range(nC):
        row = sims.getrow(r)
        if row.nnz == 0: continue
        sel = np.argsort(-row.data, kind="stable")[:ANCHORS]
        idx, s = row.indices[sel], row.data[sel]
        w = np.exp(s - s.max()); w /= w.sum()                    # beta = 0 placement, as in E6
        out[r] = (V[idx] * w[:, None]).sum(0)
    return normalize(out, axis=1)

seen = {}
for u, i in zip(train.user_id, train.item_id): seen.setdefault(int(u), set()).add(int(i))
targets = {}
for u, i in zip(heldout.user_id, heldout.item_id):
    if int(u) in umap: targets.setdefault(int(u), set()).add(int(i))
eval_users = np.array(sorted(targets)); nU = len(eval_users)
Ue = normalize(U[[umap[u] for u in eval_users]], axis=1)
known = Ue @ normalize(V, axis=1).T
for r, u in enumerate(eval_users):
    for it in seen.get(int(u), ()):
        j = cmap.get(it)
        if j is not None and j < nI: known[r, j] = -np.inf
base_cold = {cap: Ue @ cold_vectors(cap).T for cap in CAPS}
tie = np.random.default_rng(1000 + a.tie_seed).random(len(cand))
disc = 1.0 / np.log2(np.arange(2, K + 2))
rel = [np.array([cmap[i] for i in targets[int(u)]]) for u in eval_users]
warm_rel = [x[x < nI] for x in rel]; cold_rel = [x[x >= nI] for x in rel]
art_listeners = np.zeros(nC, int)
for x in cold_rel: art_listeners[x - nI] += 1

def topk(rs, idx, n):
    idx = idx[np.isfinite(rs[idx])]
    if len(idx) > n:
        thr = np.partition(rs[idx], len(idx) - n)[len(idx) - n]
        idx = idx[rs[idx] >= thr]
    o = np.lexsort((tie[idx], -rs[idx]))
    return idx[o[:n]]

def recommend(pool, sc, ind):
    o = np.lexsort((tie[pool], -sc))
    top = o[:K]
    if ind[top].sum() >= TARGET: return pool[top]
    on = o[ind[o] > 0][:TARGET]; off = o[ind[o] == 0][:K - TARGET]
    if len(on) < TARGET: return None
    sel = np.concatenate([on, off]); sel = sel[np.lexsort((tie[pool[sel]], -sc[sel]))]
    return pool[sel]

combos = [(gm, et, cp) for cp in CAPS for gm in GAMMAS for et in ETAS]
all_known = np.arange(nI)
for control, tag_id in CONTROLS.items():
    t0 = time.time()
    ind_full = (A_full[:, tmap[tag_id]].toarray().ravel() > 0).astype(np.int8)
    on_known, off_known = all_known[ind_full[:nI] > 0], all_known[ind_full[:nI] == 0]
    pools = []
    for r in range(nU):
        rs = known[r]
        pk = np.unique(np.concatenate([topk(rs, all_known, K), topk(rs, on_known, TARGET), topk(rs, off_known, K - TARGET)]))
        pools.append(np.concatenate([pk, np.arange(nI, nI + nC)]))
    C = len(combos)
    W = np.full((C, nU), np.nan, np.float32); Cr = np.full((C, nU), np.nan, np.float32)
    Rc = np.zeros((C, nU), np.float32); Nd = np.zeros((C, nU), np.float32)
    hits_art = np.zeros((C, nC), np.int32); expo = np.zeros((C, nC), np.int32)
    shown = np.zeros((C, len(cand)), bool); sat = np.zeros(C)
    for ci, (gm, et, cp) in enumerate(combos):
        cs = base_cold[cp] + gm + et * g[None, :]
        for r in range(nU):
            pool = pools[r]; nk = len(pool) - nC
            sc = np.concatenate([known[r, pool[:nk]], cs[r]])
            top = recommend(pool, sc, ind_full[pool])
            if top is None: raise RuntimeError(f"infeasible {control} user {eval_users[r]}")
            shown[ci, top] = True; sat[ci] += ind_full[top].mean()
            h = np.isin(top, rel[r]).astype(np.float32)
            Rc[ci, r] = h.sum() / len(rel[r]); Nd[ci, r] = (h * disc).sum() / disc[:min(K, len(rel[r]))].sum()
            if len(warm_rel[r]): W[ci, r] = np.isin(top, warm_rel[r]).sum() / len(warm_rel[r])
            if len(cold_rel[r]):
                ch = np.isin(cold_rel[r], top)
                Cr[ci, r] = ch.mean(); hits_art[ci, cold_rel[r][ch] - nI] += 1
            ct = top[top >= nI]; expo[ci, ct - nI] += 1
    np.savez_compressed(OUT / f"e8_{a.split}_{control}.npz", combos=np.array(combos), users=eval_users, cold=cold,
                        tag_count=m_cold, tag_tercile=terc, warm=W, coldr=Cr, recall=Rc, ndcg=Nd,
                        hits_art=hits_art, listeners=art_listeners, expo=expo,
                        distinct_total=shown.sum(1), distinct_cold=shown[:, nI:].sum(1), satisfaction=sat / nU)
    print(f"{a.split} {control}: {C} settings x {nU} users in {time.time() - t0:.0f}s", flush=True)
print("E8_DONE", a.split, flush=True)
