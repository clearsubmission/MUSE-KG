import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import normalize

ROOT = Path(".")
DATA = ROOT / "processed/lastfm"
OUT = ROOT / "results/e1_kg"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
K = 20
rng = np.random.default_rng(SEED)

train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")
test = pd.read_csv(DATA / "test.csv")
kg = pd.read_csv(DATA / "kg_triples.csv")

users = np.sort(train.user_id.unique())
items = np.sort(train.item_id.unique())

umap = {int(u): j for j, u in enumerate(users)}
imap = {int(i): j for j, i in enumerate(items)}

nu, ni = len(users), len(items)

rows = train.user_id.map(umap).to_numpy()
cols = train.item_id.map(imap).to_numpy()

X = csr_matrix(
    (np.ones(len(train), dtype=np.float32),
     (rows, cols)),
    shape=(nu, ni)
)

confidence = (
    1.0 + np.log1p(train.play_count.to_numpy())
).astype(np.float32)

Xconf = csr_matrix(
    (confidence, (rows, cols)),
    shape=(nu, ni)
)

print("Training ALS...")

als = AlternatingLeastSquares(
    factors=64,
    regularization=0.05,
    iterations=30,
    alpha=1.0,
    num_threads=4,
    random_state=SEED,
    use_gpu=False
)
als.fit(Xconf, show_progress=False)

print("Building artist-tag matrix...")

kg["artist_id"] = (
    kg["head"].str.replace("artist_", "", regex=False).astype(int)
)
kg["tag_id"] = (
    kg["tail"].str.replace("tag_", "", regex=False).astype(int)
)

kg = kg[kg.artist_id.isin(imap)].drop_duplicates(
    ["artist_id", "tag_id"]
)

tags = np.sort(kg.tag_id.unique())
tmap = {int(t): j for j, t in enumerate(tags)}

A = csr_matrix(
    (
        np.ones(len(kg), dtype=np.float32),
        (
            kg.artist_id.map(imap).to_numpy(),
            kg.tag_id.map(tmap).to_numpy()
        )
    ),
    shape=(ni, len(tags))
)

# TF-IDF-style tag reweighting:
# downweight tags associated with many artists.
df = np.asarray((A > 0).sum(axis=0)).ravel()
idf = np.log((ni + 1) / (df + 1)) + 1.0

A = A.multiply(idf).tocsr()
A = normalize(A, axis=1).tocsr()

print("Computing semantic preference scores...")

# User profiles reflect tags of artists in TRAIN only.
user_profiles = normalize(X @ A, axis=1).tocsr()
tag_scores = (user_profiles @ A.T).toarray().astype(np.float32)

# ALS full-catalog scores.
als_scores = (
    als.user_factors @ als.item_factors.T
).astype(np.float32)

# Normalize scores independently per user.
def standardize(scores):
    mu = scores.mean(axis=1, keepdims=True)
    sd = scores.std(axis=1, keepdims=True)
    return (scores - mu) / np.maximum(sd, 1e-6)

als_scores = standardize(als_scores)
tag_scores = standardize(tag_scores)

# Apply the same train-item mask for all methods.
seen_r, seen_c = X.nonzero()

def targets_from(df):
    result = {}
    for u, i in zip(df.user_id, df.item_id):
        if int(u) in umap and int(i) in imap:
            uid = umap[int(u)]
            iid = imap[int(i)]
            result.setdefault(uid, set()).add(iid)
    return result

vt = targets_from(val)
tt = targets_from(test)

def evaluate(scores, targets):
    scores = scores.copy()
    scores[seen_r, seen_c] = -np.inf

    results = []
    coverage = set()

    for uid, relevant in targets.items():
        # Exact top-K within the unobserved catalog.
        order = np.argsort(-scores[uid], kind="stable")[:K]
        coverage.update(order.tolist())

        hits = np.array(
            [int(i in relevant) for i in order],
            dtype=np.float64
        )

        recall = hits.sum() / len(relevant)

        discounts = 1.0 / np.log2(np.arange(2, K + 2))
        dcg = np.sum(hits * discounts)

        ideal = min(K, len(relevant))
        idcg = np.sum(
            1.0 / np.log2(np.arange(2, ideal + 2))
        )
        ndcg = dcg / idcg if idcg else 0.0
        results.append((recall, ndcg))

    arr = np.asarray(results)
    return {
        "Recall@20": float(arr[:, 0].mean()),
        "NDCG@20": float(arr[:, 1].mean()),
        "Coverage@20": len(coverage) / ni,
        "Users": len(results)
    }

print("Tuning hybrid weight on validation...")

weights = np.arange(0, 1.01, 0.1)
records = []

for w in weights:
    combined = (
        (1.0 - w) * als_scores + w * tag_scores
    )
    metrics = evaluate(combined, vt)
    records.append({
        "KG_weight": round(float(w), 2),
        **metrics
    })

validation = pd.DataFrame(records)
validation.to_csv(
    OUT / "validation_weights.csv", index=False
)

best_idx = validation["NDCG@20"].idxmax()
best_weight = float(validation.loc[best_idx, "KG_weight"])

print(f"Selected KG weight: {best_weight:.2f}")

models = {
    "ALS": als_scores,
    "TagKG": tag_scores,
    "ALS+KG": (
        (1.0 - best_weight) * als_scores
        + best_weight * tag_scores
    )
}

test_results = []

for name, scores in models.items():
    metrics = evaluate(scores, tt)
    test_results.append({
        "Model": name,
        **metrics
    })

results = pd.DataFrame(test_results)
results.to_csv(
    OUT / "kg_results.csv", index=False
)

with open(OUT / "config.json", "w") as f:
    json.dump({
        "seed": SEED,
        "best_weight": best_weight,
        "selection_metric": "validation NDCG@20",
        "protocol": "known-item full-catalog ranking",
        "kg_source": "training-associated artist tags",
        "graph_edges": int(len(kg)),
        "tag_features": int(len(tags))
    }, f, indent=2)

print("\n=== KG TEST RESULTS ===")
print(results.to_string(index=False))
print("\nDone.")
