from pathlib import Path
import pandas as pd

ROOT = Path(".")
DATA = ROOT / "processed/lastfm_cold"
OUT = ROOT / "results/e2_cold_start"

original = pd.read_csv(
    DATA / "artist_tags.csv"
)[["item_id", "tag_id"]].drop_duplicates()

filtered = pd.read_csv(
    DATA / "artist_tags_no_direct_leak.csv"
)[["item_id", "tag_id"]].drop_duplicates()

# Inspect unique artist-tag pairs, not raw tagging events.
records = []

for name in ["validation", "test"]:
    split = pd.read_csv(DATA / f"{name}.csv")
    cold = set(split.item_id.unique())

    a = original[original.item_id.isin(cold)]
    b = filtered[filtered.item_id.isin(cold)]

    before = a.groupby("item_id").size()
    after = b.groupby("item_id").size()

    artist = pd.DataFrame({
        "item_id": sorted(cold)
    })

    artist["tags_before"] = (
        artist.item_id.map(before).fillna(0).astype(int)
    )

    artist["tags_after"] = (
        artist.item_id.map(after).fillna(0).astype(int)
    )

    artist["tags_removed"] = (
        artist.tags_before - artist.tags_after
    )

    artist["lost_all_tags"] = (
        (artist.tags_before > 0) &
        (artist.tags_after == 0)
    )

    artist["interaction_count"] = (
        artist.item_id.map(
            split.groupby("item_id").size()
        ).fillna(0).astype(int)
    )

    artist.to_csv(
        OUT / f"{name}_metadata_filter_audit.csv",
        index=False
    )

    records.append({
        "split": name,
        "cold_artists": len(cold),
        "tag_pairs_before": len(a),
        "tag_pairs_after": len(b),
        "tag_pairs_removed": len(a) - len(b),
        "artists_tagged_before": int(
            (artist.tags_before > 0).sum()
        ),
        "artists_tagged_after": int(
            (artist.tags_after > 0).sum()
        ),
        "artists_lost_all_tags": int(
            artist.lost_all_tags.sum()
        ),
        "mean_tags_before": artist.tags_before.mean(),
        "mean_tags_after": artist.tags_after.mean(),
        "median_tags_before": artist.tags_before.median(),
        "median_tags_after": artist.tags_after.median()
    })

summary = pd.DataFrame(records)

summary.to_csv(
    OUT / "metadata_filter_audit.csv",
    index=False
)

print("\n=== METADATA FILTER AUDIT ===")
print(summary.to_string(index=False))

print("\nSaved per-artist and summary diagnostics.")
