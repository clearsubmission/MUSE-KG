import numpy as np
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
tags = pd.read_csv(DATA / "artist_tags.csv")

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

def transfer(cold_ids):
    rows = [imap[int(i)] for i in cold_ids]
    sims = (A[rows] @ known_tags.T).tocsr()
    result = []

    for j in range(len(cold_ids)):
        row = sims.getrow(j)
        if not row.nnz:
            result.append(np.zeros(V.shape[1], dtype=np.float32))
            continue

        order = np.argsort(-row.data, kind="stable")[:20]
        idx = row.indices[order]
        weights = row.data[order]

        # Same temperature=1 softmax weighting
        # used by KARMA-Rec weighted_embedding.
        w = np.exp(weights - weights.max())
        w = w / w.sum()
        result.append((V[idx] * w[:, None]).sum(axis=0))

    return np.asarray(result, dtype=np.float32)

seen = {}
for u, i in zip(train.user_id, train.item_id):
    seen.setdefault(int(u), set()).add(int(i))

def evaluate(split, scoring, catalog):
    cold = np.sort(split.item_id.unique())
    cold_v = transfer(cold)

    if catalog == "full":
        candidate_ids = np.concatenate([items, cold])
        candidate_v = np.vstack([V, cold_v])
    else:
        candidate_ids = cold
        candidate_v = cold_v

    if scoring == "cosine":
        user_v = normalize(U, axis=1)
        candidate_v = normalize(candidate_v, axis=1)
    else:
        user_v = U

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

        order = np.lexsort((candidate_ids, -row))[:20]
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

for split_name, split in [("validation", val), ("test", test)]:
    for scoring in ["dot", "cosine"]:
        for catalog in ["full", "cold_only"]:
            result = evaluate(split, scoring, catalog)
            records.append({
                "split": split_name,
                "scoring": scoring,
                "catalog": catalog,
                **result
            })

results = pd.DataFrame(records)
results.to_csv(
    OUT / "calibration_diagnostics.csv", index=False
)

print("\n=== CALIBRATION DIAGNOSTICS ===")
print(results.to_string(index=False))
