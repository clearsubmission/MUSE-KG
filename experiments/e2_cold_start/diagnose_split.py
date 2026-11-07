from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

ROOT = Path(".")
DATA = ROOT / "processed/lastfm_cold"
OUT = ROOT / "results/e2_cold_start"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")
test = pd.read_csv(DATA / "test.csv")
tags = pd.read_csv(DATA / "artist_tags.csv")

all_items = np.sort(np.unique(np.concatenate([
    train.item_id.to_numpy(),
    val.item_id.to_numpy(),
    test.item_id.to_numpy()
])))

all_tags = np.sort(tags.tag_id.unique())

imap = {int(x): i for i, x in enumerate(all_items)}
tmap = {int(x): i for i, x in enumerate(all_tags)}

# Exclude tag records for artists outside the experiment.
n_before = len(tags)

tags = tags[
    tags.item_id.isin(imap) &
    tags.tag_id.isin(tmap)
].copy()

tags = tags.dropna(subset=["item_id", "tag_id"])

print("Original tag edges:", n_before)
print("Retained tag edges:", len(tags))
print("Excluded tag edges:", n_before - len(tags))

tag_rows = tags.item_id.map(imap).to_numpy(dtype=np.int64)
tag_cols = tags.tag_id.map(tmap).to_numpy(dtype=np.int64)

assert np.all(tag_rows >= 0)
assert np.all(tag_cols >= 0)

A = csr_matrix(
    (
        np.ones(len(tags), dtype=np.float32),
        (tag_rows, tag_cols)
    ),
    shape=(len(all_items), len(all_tags))
)

A.data[:] = 1.0

train_idx = np.array([
    imap[int(i)] for i in train.item_id.unique()
])

# Train-derived IDF only
df = np.asarray((A[train_idx] > 0).sum(axis=0)).ravel()
idf = np.log((len(train_idx) + 1) / (df + 1)) + 1.0

A = normalize(A.multiply(idf).tocsr(), axis=1)
train_A = A[train_idx]

records = []

for name, split in [
    ("validation", val),
    ("test", test)
]:
    items = np.sort(split.item_id.unique())
    idx = np.array([imap[int(i)] for i in items])

    S = (A[idx] @ train_A.T).tocsr()

    max_sim = S.max(axis=1).toarray().ravel()
    tag_count = np.diff(A[idx].indptr)

    counts = split.groupby("item_id").size()
    positives = split.groupby("user_id").size()

    result = {
        "split": name,
        "interactions": len(split),
        "artists": len(items),
        "users": split.user_id.nunique(),
        "mean_interactions_per_artist": counts.mean(),
        "median_interactions_per_artist": counts.median(),
        "max_interactions_per_artist": counts.max(),
        "mean_positives_per_user": positives.mean(),
        "median_positives_per_user": positives.median(),
        "tagged_artists": int((tag_count > 0).sum()),
        "mean_tags_per_artist": float(tag_count.mean()),
        "median_tags_per_artist": float(np.median(tag_count)),
        "artists_with_anchor": int((max_sim > 0).sum()),
        "mean_max_anchor_similarity": float(max_sim.mean()),
        "median_max_anchor_similarity": float(np.median(max_sim)),
        "p90_max_anchor_similarity": float(
            np.percentile(max_sim, 90)
        )
    }

    records.append(result)

    artist_df = pd.DataFrame({
        "item_id": items,
        "tag_count": tag_count,
        "max_anchor_similarity": max_sim,
        "interaction_count": [
            int(counts.get(i, 0)) for i in items
        ]
    })

    artist_df.to_csv(
        OUT / f"{name}_artist_diagnostics.csv",
        index=False
    )

summary = pd.DataFrame(records)
summary.to_csv(
    OUT / "split_diagnostics.csv",
    index=False
)

print("\n=== SPLIT DIAGNOSTICS ===")
print(summary.T.to_string(header=False))

print("\nSaved diagnostic files.")
