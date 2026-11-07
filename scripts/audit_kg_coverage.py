from pathlib import Path
import pandas as pd
import numpy as np

root = Path(".")

train = pd.read_csv(root / "processed/lastfm/train.csv")
kg = pd.read_csv(root / "processed/lastfm/kg_triples.csv")

kg_artists = set(
    kg["head"].str.replace("artist_", "", regex=False).astype(int)
)

train_artists = set(train["item_id"].unique())

covered = train_artists & kg_artists
missing = train_artists - kg_artists

print("=== KNOWLEDGE GRAPH COVERAGE ===")
print("Training artists:", len(train_artists))
print("Artists with KG edges:", len(covered))
print("Artists without KG edges:", len(missing))
print("Coverage (%):", round(100 * len(covered) / len(train_artists), 2))

counts = train.groupby("item_id").size()

for minimum in [1, 2, 5, 10, 20]:
    subset = set(counts[counts >= minimum].index)
    count = len(subset & kg_artists)

    print(
        f"Artists with >= {minimum} interactions: "
        f"{len(subset)} total, "
        f"{count} KG-covered "
        f"({100 * count / len(subset):.2f}%)"
    )

print("\nTag frequency distribution:")
print(kg.groupby("tail").size().describe())

print("\nAudit complete.")
