from pathlib import Path
import numpy as np
import pandas as pd
import json

ROOT = Path(".")
RAW = ROOT / "data/lastfm"
OUT = ROOT / "processed/lastfm_cold"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

df = pd.read_csv(
    RAW / "user_artists.dat",
    sep="\t"
).rename(columns={
    "userID": "user_id",
    "artistID": "item_id",
    "weight": "play_count"
})

user_tags = pd.read_csv(
    RAW / "user_taggedartists.dat",
    sep="\t"
).rename(columns={
    "userID": "user_id",
    "artistID": "item_id",
    "tagID": "tag_id"
})

# Select artists with enough interactions to
# measure cold-start ranking reliably.
counts = df.groupby("item_id").size()
eligible = np.sort(counts[counts >= 5].index.to_numpy())

rng.shuffle(eligible)

n_test = max(1, int(0.10 * len(eligible)))
n_val = max(1, int(0.10 * len(eligible)))

test_items = set(eligible[:n_test])
val_items = set(eligible[n_test:n_test + n_val])

train_items = set(df.item_id) - test_items - val_items

train = df[df.item_id.isin(train_items)].copy()
val = df[df.item_id.isin(val_items)].copy()
test = df[df.item_id.isin(test_items)].copy()

# Evaluate only users with training history.
train_users = set(train.user_id)

val = val[val.user_id.isin(train_users)].copy()
test = test[test.user_id.isin(train_users)].copy()

assert not (set(train.item_id) & set(val.item_id))
assert not (set(train.item_id) & set(test.item_id))
assert not (set(val.item_id) & set(test.item_id))

# Tag metadata from the separate user-tag file is
# allowed only under a declared side-information protocol.
# It must be independent of held-out recommendation labels.
tag_edges = user_tags[
    ["item_id", "tag_id"]
].drop_duplicates()

for name, data in [
    ("train", train),
    ("validation", val),
    ("test", test)
]:
    data.to_csv(OUT / f"{name}.csv", index=False)

tag_edges.to_csv(OUT / "artist_tags.csv", index=False)

summary = {
    "seed": SEED,
    "split": "disjoint artist holdout",
    "train_interactions": len(train),
    "validation_interactions": len(val),
    "test_interactions": len(test),
    "train_artists": train.item_id.nunique(),
    "validation_artists": val.item_id.nunique(),
    "test_artists": test.item_id.nunique(),
    "train_users": train.user_id.nunique(),
    "tagged_test_artists": len(
        set(test.item_id) & set(tag_edges.item_id)
    ),
    "tagged_validation_artists": len(
        set(val.item_id) & set(tag_edges.item_id)
    )
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("\nCold-start split completed.")
