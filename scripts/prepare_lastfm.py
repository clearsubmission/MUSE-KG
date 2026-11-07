from pathlib import Path
import numpy as np
import pandas as pd
import json

ROOT = Path(".")
RAW = ROOT / "data/lastfm"
OUT = ROOT / "processed/lastfm"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

print("Loading Last.fm data...")

interactions = pd.read_csv(
    RAW / "user_artists.dat", sep="\t"
).rename(columns={
    "userID": "user_id",
    "artistID": "item_id",
    "weight": "play_count"
})

artists = pd.read_csv(
    RAW / "artists.dat",
    sep="\t", encoding="latin-1"
).rename(columns={"id": "item_id"})

tags = pd.read_csv(
    RAW / "tags.dat",
    sep="\t", encoding="latin-1"
)

user_tags = pd.read_csv(
    RAW / "user_taggedartists.dat",
    sep="\t"
).rename(columns={
    "userID": "user_id",
    "artistID": "item_id",
    "tagID": "tag_id"
})

print("Building user-level splits...")

# Users need at least three interactions for
# train / validation / test splitting.
counts = interactions.groupby("user_id").size()
eligible = counts[counts >= 3].index

interactions = interactions[
    interactions.user_id.isin(eligible)
].copy()

train_parts = []
val_parts = []
test_parts = []

for user_id, group in interactions.groupby("user_id", sort=True):
    indices = rng.permutation(len(group))

    test_idx = indices[0]
    val_idx = indices[1]
    train_idx = indices[2:]

    test_parts.append(group.iloc[[test_idx]])
    val_parts.append(group.iloc[[val_idx]])
    train_parts.append(group.iloc[train_idx])

train = pd.concat(train_parts, ignore_index=True)
val = pd.concat(val_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

# This is a known-item recommendation split.
# Items absent from training are not evaluated here.
known_items = set(train.item_id)

val = val[val.item_id.isin(known_items)].copy()
test = test[test.item_id.isin(known_items)].copy()

for name, df in [
    ("train", train),
    ("validation", val),
    ("test", test)
]:
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name}: {len(df):,} interactions")

print("Building artist-tag graph...")

# Aggregate artist tags only from users who have
# a corresponding TRAIN interaction with the artist.
# This conservative rule prevents held-out
# user-item tagging evidence from leaking into KG.
tagging = user_tags.merge(
    train[["user_id", "item_id"]],
    on=["user_id", "item_id"],
    how="inner"
)

artist_tags = tagging[
    ["item_id", "tag_id"]
].drop_duplicates()

kg = pd.DataFrame({
    "head": "artist_" + artist_tags.item_id.astype(str),
    "relation": "has_tag",
    "tail": "tag_" + artist_tags.tag_id.astype(str)
})

kg.to_csv(OUT / "kg_triples.csv", index=False)
artists.to_csv(OUT / "artists.csv", index=False)
tags.to_csv(OUT / "tags.csv", index=False)

summary = {
    "seed": SEED,
    "split_type": "user-wise random leave-two-out",
    "evaluation_type": "known-item recommendation",
    "train_interactions": len(train),
    "validation_interactions": len(val),
    "test_interactions": len(test),
    "users": int(interactions.user_id.nunique()),
    "train_items": int(train.item_id.nunique()),
    "kg_triples": len(kg)
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSummary:")
print(json.dumps(summary, indent=2))
print("\nPreprocessing complete.")
