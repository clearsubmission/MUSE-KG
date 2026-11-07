from pathlib import Path
import pandas as pd

ROOT = Path(".")

print("\n=== LAST.FM ===")

lf = ROOT / "data/lastfm"

files = {
    "interactions": "user_artists.dat",
    "artists": "artists.dat",
    "tags": "tags.dat",
    "user_tags": "user_taggedartists.dat",
    "friends": "user_friends.dat",
}

for name, filename in files.items():
    path = lf / filename

    if not path.exists():
        print(f"MISSING: {path}")
        continue

    df = pd.read_csv(path, sep="\t", encoding="latin-1")

    print(f"\n{name}: {len(df):,} rows")
    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string(index=False))

print("\n=== MTG-JAMENDO ===")

jm = ROOT / "data/jamendo/mtg-jamendo-dataset/data"

for filename in [
    "autotagging.tsv",
    "autotagging_genre.tsv",
    "autotagging_instrument.tsv",
    "autotagging_moodtheme.tsv",
]:
    path = jm / filename

    if not path.exists():
        print(f"MISSING: {path}")
        continue

    with path.open(encoding="utf-8") as f:
        rows = sum(1 for _ in f) - 1

    print(f"{filename}: {rows:,} records")

print("\nAudit complete.")
