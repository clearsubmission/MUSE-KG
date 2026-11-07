from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
DATA = ROOT / "processed/lastfm_cold"

original = pd.read_csv(
    DATA / "artist_tags.csv"
)[["item_id", "tag_id"]].drop_duplicates()

filtered = pd.read_csv(
    DATA / "artist_tags_no_direct_leak.csv"
)[["item_id", "tag_id"]].drop_duplicates()

retained_counts = (
    filtered.groupby("item_id").size().to_dict()
)

for seed in [11, 22, 33]:
    rng = np.random.default_rng(seed)
    parts = []

    for artist, group in original.groupby("item_id"):
        n = retained_counts.get(artist, 0)

        if n == 0:
            continue

        if n > len(group):
            raise ValueError(
                f"Filtered count exceeds original for artist {artist}"
            )

        chosen = rng.choice(
            len(group), size=n, replace=False
        )

        parts.append(group.iloc[chosen])

    result = pd.concat(parts, ignore_index=True)

    output = DATA / f"artist_tags_matched_seed{seed}.csv"
    result.to_csv(output, index=False)

    print(
        f"Seed {seed}: {len(result)} tag pairs saved "
        f"to {output}"
    )

print("Matched random tag controls completed.")
