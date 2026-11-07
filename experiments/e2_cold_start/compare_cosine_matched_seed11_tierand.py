import numpy as np, os
_TIE_SEED = int(os.environ.get('TIE_SEED', 0))
_TIE = np.random.default_rng(1000 + _TIE_SEED)
import pandas as pd
import torch
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

ROOT = Path(".")
DATA = ROOT / "processed/lastfm_cold"
OUT = ROOT / "results/e2_cold_start"

ckpt = torch.load(
    OUT / "mf_bpr_checkpoint.pt",
    map_location="cpu",
    weights_only=False
)

U = ckpt["model_state"]["user_emb.weight"].numpy()
V = ckpt["model_state"]["item_emb.weight"].numpy()

users = np.asarray(ckpt["users"], dtype=int)
items = np.asarray(ckpt["items"], dtype=int)

umap = {int(x): j for j, x in enumerate(users)}

train = pd.read_csv(DATA / "train.csv")
tags = pd.read_csv(DATA / "artist_tags_matched_seed11.csv")

val = pd.read_csv(DATA / "validation.csv")
test = pd.read_csv(DATA / "test.csv")

all_items = np.sort(np.unique(np.concatenate([
    items, val.item_id.unique(), test.item_id.unique()
])))

imap = {int(x): j for j, x in enumerate(all_items)}
tmap = {int(x): j for j, x in enumerate(
    sorted(tags.tag_id.unique())
)}

tags = tags[tags.item_id.isin(imap)].copy()

A = csr_matrix(
    (
        np.ones(len(tags), dtype=np.float32),
        (
            tags.item_id.map(imap).to_numpy(dtype=int),
            tags.tag_id.map(tmap).to_numpy(dtype=int)
        )
    ),
    shape=(len(all_items), len(tmap))
)
A.data[:] = 1.0

known_rows = np.array([imap[int(i)] for i in items])

df = np.asarray((A[known_rows] > 0).sum(axis=0)).ravel()
idf = np.log((len(items) + 1) / (df + 1)) + 1

A = normalize(A.multiply(idf).tocsr(), axis=1)
known_tags = A[known_rows]

def transfer(cold_ids, mode):
    rows = [imap[int(i)] for i in cold_ids]
    sims = (A[rows] @ known_tags.T).tocsr()
    result = []

    for j in range(len(cold_ids)):
        row = sims.getrow(j)

        if mode == "random":
            rng = np.random.default_rng(42 + int(cold_ids[j]))
            idx = rng.choice(
                len(items), size=min(20, len(items)),
                replace=False
            )
            result.append(V[idx].mean(axis=0))
            continue

        if mode == "zero":
            result.append(
                np.zeros(V.shape[1], dtype=np.float32)
            )
            continue

        if not row.nnz:
            result.append(np.zeros(V.shape[1], dtype=np.float32))
            continue

        order = np.lexsort((_TIE.random(len(row.data)), -row.data))[:20]
        idx = row.indices[order]
        weights = row.data[order]

        if mode == "semantic":
            result.append(V[idx].mean(axis=0))

        elif mode == "weighted":
            w = np.exp(weights - weights.max())
            w = w / w.sum()
            result.append((V[idx] * w[:, None]).sum(axis=0))

        else:
            raise ValueError(mode)

    return np.asarray(result, dtype=np.float32)

seen = {}
for u, i in zip(train.user_id, train.item_id):
    seen.setdefault(int(u), set()).add(int(i))

def evaluate(split, scoring, catalog, mode):
    cold = np.sort(split.item_id.unique())
    cold_v = transfer(cold, mode)

    if catalog == "full":
        candidate_ids = np.concatenate([items, cold])
        candidate_v = np.vstack([V, cold_v])
    else:
        candidate_ids = cold
        candidate_v = cold_v

    if scoring == "cosine":
        user_v = normalize(U, axis=1)
        candidate_v = normalize(candidate_v, axis=1)

    elif scoring == "norm_matched":
        user_v = U

        known_norm = np.linalg.norm(V, axis=1).mean()

        cold_norm = np.linalg.norm(
            cold_v, axis=1, keepdims=True
        )

        adjusted_cold = (
            cold_v / np.maximum(cold_norm, 1e-12)
        ) * known_norm

        if catalog == "full":
            candidate_v = np.vstack([V, adjusted_cold])
        else:
            candidate_v = adjusted_cold

    elif scoring == "dot":
        user_v = U

    else:
        raise ValueError(scoring)

    scores = user_v @ candidate_v.T
    cmap = {int(x): j for j, x in enumerate(candidate_ids)}

    targets = {}
    for u, i in zip(split.user_id, split.item_id):
        targets.setdefault(int(u), set()).add(int(i))

    recalls, ndcgs = [], []
    recovered = set()

    for u, relevant in targets.items():
        if u not in umap:
            continue

        row = scores[umap[u]].copy()

        if catalog == "full":
            for i in seen.get(u, set()):
                if i in cmap:
                    row[cmap[i]] = -np.inf

        order = np.lexsort((_TIE.random(len(candidate_ids)), -row))[:20]
        recommended = [int(candidate_ids[j]) for j in order]

        hits = np.array(
            [int(i in relevant) for i in recommended],
            dtype=float
        )

        recovered.update(set(recommended) & relevant)

        discounts = 1 / np.log2(np.arange(2, 22))
        dcg = np.sum(hits * discounts)
        idcg = np.sum(discounts[:min(20, len(relevant))])

        recalls.append(hits.sum() / len(relevant))
        ndcgs.append(dcg / idcg if idcg else 0)

    return {
        "Recall@20": np.mean(recalls),
        "NDCG@20": np.mean(ndcgs),
        "RecoveredArtists": len(recovered),
        "ColdArtists": len(cold),
        "KnownEmbeddingNorm": float(
            np.linalg.norm(V, axis=1).mean()
        ),
        "ColdEmbeddingNorm": float(
            np.linalg.norm(cold_v, axis=1).mean()
        )
    }


records = []

for split_name, split in [
    ("validation", val),
    ("test", test)
]:
    for mode in [
        "zero", "random", "semantic", "weighted"
    ]:
        for catalog in ["full", "cold_only"]:
            result = evaluate(
                split, "cosine", catalog, mode
            )
            records.append({
                "split": split_name,
                "anchor_method": mode,
                "catalog": catalog,
                **result
            })

results = pd.DataFrame(records)

output = OUT / "cosine_anchor_matched_seed11_tierand{_TIE_SEED}.csv"
results.to_csv(output, index=False)

print("\n=== COSINE ANCHOR COMPARISON ===")
print(results.to_string(index=False))
print("\nSaved:", output)
