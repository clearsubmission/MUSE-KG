from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")
OUT = ROOT / "results/e2_cold_start"

records = []

for split in ["validation", "test"]:
    for mode in ["random", "semantic", "weighted"]:

        path = OUT / f"{split}_{mode}_predictions.csv"
        p = pd.read_csv(path)

        artist = p.groupby("item_id").agg(
            positives=("hit_at_20", "size"),
            hits=("hit_at_20", "sum")
        )

        artist["hit_rate"] = (
            artist.hits / artist.positives
        )

        hits = int(p.hit_at_20.sum())
        total = len(p)

        top_artist = int(artist.hits.idxmax())

        remaining = p[p.item_id != top_artist]

        records.append({
            "split": split,
            "method": mode,
            "interaction_hit_rate": hits / total,
            "artist_macro_hit_rate": artist.hit_rate.mean(),
            "recovered_artists": int(
                (artist.hits > 0).sum()
            ),
            "total_artists": len(artist),
            "leading_artist_id": top_artist,
            "leading_artist_hits": int(
                artist.loc[top_artist, "hits"]
            ),
            "hit_rate_without_leader": (
                remaining.hit_at_20.mean()
                if len(remaining) else np.nan
            )
        })

results = pd.DataFrame(records)

results.to_csv(
    OUT / "retrieval_robustness.csv",
    index=False
)

print(results.to_string(index=False))
