from pathlib import Path
import pandas as pd

ROOT = Path(".")
DATA = ROOT / "processed/lastfm_cold"
RAW = ROOT / "data/lastfm"

train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")
test = pd.read_csv(DATA / "test.csv")

tags = pd.read_csv(
    RAW / "user_taggedartists.dat",
    sep="\t"
).rename(columns={
    "userID": "user_id",
    "artistID": "item_id",
    "tagID": "tag_id"
})

# All held-out user-artist interactions.
heldout = pd.concat([
    val[["user_id", "item_id"]],
    test[["user_id", "item_id"]]
]).drop_duplicates()

# Find tagging records associated with held-out pairs.
merged = tags.merge(
    heldout.assign(_heldout=1),
    on=["user_id", "item_id"],
    how="left"
)

direct_leakage = merged["_heldout"].eq(1)

print("Original tagging records:", len(tags))
print("Records tied to held-out pairs:",
      int(direct_leakage.sum()))

print("Direct contamination rate:",
      round(100 * direct_leakage.mean(), 3), "%")

# Remove tagging records contributed by users whose
# matching user-artist interaction is held out.
clean = merged.loc[
    ~direct_leakage,
    ["item_id", "tag_id"]
].drop_duplicates()

clean.to_csv(
    DATA / "artist_tags_no_direct_leak.csv",
    index=False
)

for name, split in [
    ("validation", val),
    ("test", test)
]:
    covered = len(
        set(split.item_id) & set(clean.item_id)
    )

    print(
        f"{name}: {covered}/{split.item_id.nunique()} "
        "artists retain tags"
    )

print("Saved sanitized artist metadata.")
