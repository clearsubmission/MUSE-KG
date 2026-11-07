# Who Gets Discovered? Code for the EAIM 2027 submission

Audit of tag-based cold-start music recommendation on HetRec 2011 Last.fm.

## Data
Download the HetRec 2011 Last.fm dataset (hetrec2011-lastfm-2k) from GroupLens and unzip it to `data/lastfm/`.

## Environment
Python 3.11 with numpy, pandas, scipy, scikit-learn, implicit and torch (2.x).

## Run order
1. `scripts/` – data audit and preprocessing (`prepare_lastfm.py`, `audit_data.py`, `audit_kg_coverage.py`)
2. `experiments/e1_*` – warm-start accuracy (ALS, ItemKNN, tag hybrid)
3. `experiments/e2_cold_start/` – cold-start split, tag sanitization, anchor placement, tier analysis, tag-cap and quota interventions, ridge baseline
4. `experiments/e3_controllability/`, `experiments/e4_control_transfer/` – genre steering and shuffled-label controls
5. `experiments/e5_mixed_relevance/` – discovery bonus (E5/E6) and bootstrap budget controller (E7)

Seeds are fixed in each script (split seed 42). Results are written to `results/`.
