from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")
DATA = ROOT / "processed/lastfm_cold"
OUT = ROOT / "results/e2_cold_start"

artists = pd.read_csv(
    ROOT / "data/lastfm/artists.dat",
    sep="\t",
    encoding="latin-1"
)[["id", "name"]].rename(columns={"id": "item_id"})

records = []

for name in ["validation", "test"]:
    df = pd.read_csv(DATA / f"{name}.csv")

    counts = (
        df.groupby("item_id")
          .agg(
              interactions=("user_id", "size"),
              unique_users=("user_id", "nunique")
          )
          .reset_index()
          .sort_values("interactions", ascending=False)
    )

    counts = counts.merge(artists, on="item_id", how="left")
    counts.to_csv(OUT / f"{name}_artist_popularity.csv", index=False)

    total = len(df)

    print(f"\n=== {name.upper()} ===")

    for k in [1, 5, 10, 20, 50]:
        share = counts.head(k).interactions.sum() / total
        print(f"Top {k} artist interaction share: {share:.4f}")

    print("\nTop 10 artists:")
    print(
        counts[["item_id", "name", "interactions"]]
        .head(10)
        .to_string(index=False)
    )

    records.append({
        "split": name,
        "top1_share": counts.head(1).interactions.sum()/total,
        "top5_share": counts.head(5).interactions.sum()/total,
        "top10_share": counts.head(10).interactions.sum()/total,
        "top20_share": counts.head(20).interactions.sum()/total,
        "top50_share": counts.head(50).interactions.sum()/total,
        "interaction_count": total
    })

pd.DataFrame(records).to_csv(
    OUT / "artist_concentration.csv",
    index=False
)

print("\nAudit complete.")
