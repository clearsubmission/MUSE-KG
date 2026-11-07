from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(".")
SOURCE = ROOT / "data/lastfm/user_artists.dat"
OUT = ROOT / "processed/e5_mixed_relevance"

OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_COLD_VAL = 282
N_COLD_TEST = 282
MIN_COLD_INTERACTIONS = 5
WARM_VAL_FRACTION = 0.10
WARM_TEST_FRACTION = 0.10

rng = np.random.default_rng(SEED)

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

raw = pd.read_csv(SOURCE, sep="\t")

print("Original columns:", raw.columns.tolist())

raw = raw.rename(columns={
    "userID": "user_id",
    "artistID": "item_id",
})

required = {"user_id", "item_id"}
assert required.issubset(raw.columns)

df = raw[["user_id", "item_id"]].copy()

df = df.dropna().drop_duplicates()
df = df.astype({"user_id": int, "item_id": int})

# Select eligible cold artists before partitioning.
artist_counts = df.groupby("item_id").size()
eligible = artist_counts[
    artist_counts >= MIN_COLD_INTERACTIONS
].index.to_numpy()

if len(eligible) < N_COLD_VAL + N_COLD_TEST:
    raise RuntimeError("Not enough eligible cold artists")

selected = rng.choice(
    eligible,
    size=N_COLD_VAL + N_COLD_TEST,
    replace=False
)

cold_val = set(map(int, selected[:N_COLD_VAL]))
cold_test = set(map(int, selected[N_COLD_VAL:]))

is_cold_val = df.item_id.isin(cold_val)
is_cold_test = df.item_id.isin(cold_test)

cold_validation = df[is_cold_val].copy()
cold_test_data = df[is_cold_test].copy()

known = df[~(is_cold_val | is_cold_test)].copy()

# Split known-artist interactions per user.
# Preserve at least one known interaction in train
# for every user with known interactions.
train_parts = []
warm_val_parts = []
warm_test_parts = []

for user, group in known.groupby("user_id", sort=True):
    indices = rng.permutation(len(group))

    n = len(group)

    if n < 3:
        n_val = 0
        n_test = 0
    else:
        n_val = int(np.floor(n * WARM_VAL_FRACTION))
        n_test = int(np.floor(n * WARM_TEST_FRACTION))

        if n >= 10:
            n_val = max(1, n_val)
            n_test = max(1, n_test)

        while n_val + n_test >= n:
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1

    warm_val_parts.append(
        group.iloc[indices[:n_val]]
    )
    warm_test_parts.append(
        group.iloc[indices[n_val:n_val+n_test]]
    )
    train_parts.append(
        group.iloc[indices[n_val+n_test:]]
    )

train = pd.concat(train_parts, ignore_index=True)
warm_val = pd.concat(warm_val_parts, ignore_index=True)
warm_test = pd.concat(warm_test_parts, ignore_index=True)

# Only include test/validation records for users
# that can be represented by the trained model.
train_users = set(train.user_id)

cold_validation = cold_validation[
    cold_validation.user_id.isin(train_users)
].copy()

cold_test_data = cold_test_data[
    cold_test_data.user_id.isin(train_users)
].copy()

validation = pd.concat(
    [warm_val, cold_validation],
    ignore_index=True
)

test = pd.concat(
    [warm_test, cold_test_data],
    ignore_index=True
)

train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)
test = test.reset_index(drop=True)

train_items = set(train.item_id)
val_cold_items = set(cold_validation.item_id)
test_cold_items = set(cold_test_data.item_id)

assert train_items.isdisjoint(val_cold_items)
assert train_items.isdisjoint(test_cold_items)
assert val_cold_items.isdisjoint(test_cold_items)

def pairset(x):
    return set(map(tuple, x[["user_id", "item_id"]].to_numpy()))

assert pairset(train).isdisjoint(pairset(validation))
assert pairset(train).isdisjoint(pairset(test))
assert pairset(validation).isdisjoint(pairset(test))

assert set(validation.user_id).issubset(train_users)
assert set(test.user_id).issubset(train_users)

# Ensure all known-artist evaluation items are
# represented in the training artist catalog.
warm_val = warm_val[
    warm_val.item_id.isin(train_items)
].copy()

warm_test = warm_test[
    warm_test.item_id.isin(train_items)
].copy()

# Rebuild combined sets after catalog validation.
validation = pd.concat(
    [warm_val, cold_validation],
    ignore_index=True
)

test = pd.concat(
    [warm_test, cold_test_data],
    ignore_index=True
)

assert pairset(train).isdisjoint(pairset(validation))
assert pairset(train).isdisjoint(pairset(test))
assert pairset(validation).isdisjoint(pairset(test))

assert set(warm_val.item_id).issubset(train_items)
assert set(warm_test.item_id).issubset(train_items)

train.to_csv(OUT / "train.csv", index=False)
validation.to_csv(OUT / "validation.csv", index=False)
test.to_csv(OUT / "test.csv", index=False)

warm_val.to_csv(OUT / "warm_validation.csv", index=False)
warm_test.to_csv(OUT / "warm_test.csv", index=False)
cold_validation.to_csv(
    OUT / "cold_validation.csv", index=False
)
cold_test_data.to_csv(
    OUT / "cold_test.csv", index=False
)

manifest = {
    "seed": SEED,
    "source": str(SOURCE),
    "train_interactions": len(train),
    "warm_validation_interactions": len(warm_val),
    "cold_validation_interactions": len(cold_validation),
    "warm_test_interactions": len(warm_test),
    "cold_test_interactions": len(cold_test_data),
    "train_users": len(train_users),
    "known_train_artists": len(train_items),
    "cold_validation_artists": len(val_cold_items),
    "cold_test_artists": len(test_cold_items),
    "warm_validation_fraction": WARM_VAL_FRACTION,
    "warm_test_fraction": WARM_TEST_FRACTION,
    "minimum_cold_artist_interactions": MIN_COLD_INTERACTIONS,
}

(OUT / "split_manifest.json").write_text(
    json.dumps(manifest, indent=2)
)

print("\n=== E5 MIXED-RELEVANCE SPLIT ===")
for key, value in manifest.items():
    print(f"{key}: {value}")

print("\nPair-disjoint and artist-disjoint checks passed.")
print("Saved to:", OUT)
