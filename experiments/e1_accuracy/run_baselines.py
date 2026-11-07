import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender

ROOT = Path(".")
DATA = ROOT / "processed/lastfm"
OUT = ROOT / "results/e1_accuracy"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TOPK = (10, 20)
MAX_K = max(TOPK)

np.random.seed(SEED)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("Loading train, validation, and test datasets...")

train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")
test = pd.read_csv(DATA / "test.csv")

users = np.sort(train.user_id.unique())
items = np.sort(train.item_id.unique())

user_map = {int(u): i for i, u in enumerate(users)}
item_map = {int(a): i for i, a in enumerate(items)}

n_users = len(users)
n_items = len(items)

def build_matrix(df, weighted=False):
    valid = df[
        df.user_id.isin(user_map) &
        df.item_id.isin(item_map)
    ].copy()

    rows = valid.user_id.map(user_map).to_numpy()
    cols = valid.item_id.map(item_map).to_numpy()

    if weighted:
        values = (
            1.0 + np.log1p(valid.play_count.to_numpy())
        ).astype(np.float32)
    else:
        values = np.ones(len(valid), dtype=np.float32)

    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    matrix.sum_duplicates()
    return matrix

X = build_matrix(train, weighted=False)
X_conf = build_matrix(train, weighted=True)

print(f"Users: {n_users:,}")
print(f"Items: {n_items:,}")
print(f"Training interactions: {X.nnz:,}")

# --------------------------------------------------
# 2. EVALUATION
# --------------------------------------------------

def make_targets(df):
    targets = {}
    for u, i in zip(df.user_id, df.item_id):
        if int(u) in user_map and int(i) in item_map:
            uid = user_map[int(u)]
            iid = item_map[int(i)]
            targets.setdefault(uid, set()).add(iid)
    return targets

val_targets = make_targets(val)
test_targets = make_targets(test)

def evaluate(recommend_fn, targets):
    records = []
    covered = set()

    for uid, relevant in targets.items():
        if not relevant:
            continue

        recommended = np.asarray(
            recommend_fn(uid), dtype=np.int64
        )[:MAX_K]

        if len(recommended) != len(set(recommended)):
            raise ValueError("Duplicate recommendations detected")

        seen = set(X[uid].indices)
        if seen.intersection(recommended):
            raise ValueError("Training items leaked into ranking")

        for k in TOPK:
            ranking = recommended[:k]
            hits = np.array(
                [int(i in relevant) for i in ranking],
                dtype=np.float64
            )

            recall = hits.sum() / len(relevant)

            discounts = 1.0 / np.log2(
                np.arange(2, len(ranking) + 2)
            )
            dcg = np.sum(hits * discounts)

            ideal_length = min(k, len(relevant))
            idcg = np.sum(
                1.0 / np.log2(
                    np.arange(2, ideal_length + 2)
                )
            )

            ndcg = dcg / idcg if idcg > 0 else 0.0

            records.append({
                "user": uid,
                "k": k,
                "recall": recall,
                "ndcg": ndcg
            })

        covered.update(recommended)

    df = pd.DataFrame(records)
    result = {}

    for k in TOPK:
        subset = df[df.k == k]
        result[f"Recall@{k}"] = float(
            subset.recall.mean()
        )
        result[f"NDCG@{k}"] = float(
            subset.ndcg.mean()
        )

    result["Coverage@20"] = len(covered) / n_items
    result["EvaluatedUsers"] = len(targets)

    return result

# --------------------------------------------------
# 3. MOST POPULAR
# --------------------------------------------------

print("\nTraining MostPopular...")

popularity = np.asarray(
    X.sum(axis=0)
).ravel()

popular_order = np.argsort(-popularity, kind="stable")

def recommend_popular(uid):
    seen = set(X[uid].indices)

    return [
        int(i) for i in popular_order
        if i not in seen
    ][:MAX_K]

# --------------------------------------------------
# 4. ITEM KNN
# --------------------------------------------------

print("Training ItemKNN...")

start = time.perf_counter()

knn = CosineRecommender(K=100)
knn.fit(X, show_progress=True)

knn_time = time.perf_counter() - start

def recommend_knn(uid):
    # Retrieve all candidates to guarantee 20 unseen items.
    # implicit's neighborhood recommender may return seen items.
    scores = knn.recommend(
        uid,
        X[uid],
        N=n_items,
        filter_already_liked_items=False
    )

    ids = scores[0]
    seen = set(X[uid].indices)

    return [
        int(i) for i in ids
        if int(i) not in seen
    ][:MAX_K]

# --------------------------------------------------
# 5. IMPLICIT ALS
# --------------------------------------------------

print("Training Implicit ALS...")

start = time.perf_counter()

als = AlternatingLeastSquares(
    factors=64,
    regularization=0.05,
    iterations=30,
    alpha=1.0,
    random_state=SEED,
    num_threads=4,
    use_gpu=False
)

als.fit(X_conf, show_progress=True)

als_time = time.perf_counter() - start

def recommend_als(uid):
    ids, _ = als.recommend(
        uid,
        X_conf[uid],
        N=n_items,
        filter_already_liked_items=False
    )

    seen = set(X[uid].indices)

    return [
        int(i) for i in ids
        if int(i) not in seen
    ][:MAX_K]

# --------------------------------------------------
# 6. RESULTS
# --------------------------------------------------

models = {
    "MostPopular": recommend_popular,
    "ItemKNN": recommend_knn,
    "ImplicitALS": recommend_als
}

rows = []

for name, fn in models.items():
    print(f"\nEvaluating {name}...")

    validation = evaluate(fn, val_targets)
    testing = evaluate(fn, test_targets)

    result = {
        "Model": name,
        "TrainSeconds": {
            "MostPopular": 0.0,
            "ItemKNN": knn_time,
            "ImplicitALS": als_time
        }[name]
    }

    for metric, value in validation.items():
        result[f"val_{metric}"] = value

    for metric, value in testing.items():
        result[f"test_{metric}"] = value

    rows.append(result)

results = pd.DataFrame(rows)

results.to_csv(
    OUT / "baseline_results.csv",
    index=False
)

print("\n=== TEST RESULTS ===")

columns = [
    "Model",
    "test_Recall@10",
    "test_Recall@20",
    "test_NDCG@10",
    "test_NDCG@20",
    "test_Coverage@20",
    "TrainSeconds"
]

print(results[columns].to_string(index=False))

with open(OUT / "experiment_config.json", "w") as f:
    json.dump({
        "seed": SEED,
        "dataset": "Last.fm HetRec 2011",
        "protocol": "known-item, full-catalog ranking",
        "candidate_items": n_items,
        "training_users": n_users,
        "knn_neighbors": 100,
        "als_factors": 64,
        "als_iterations": 30,
        "als_regularization": 0.05,
        "als_confidence": "1 + log1p(play_count)"
    }, f, indent=2)

print("\nResults saved to:", OUT)
