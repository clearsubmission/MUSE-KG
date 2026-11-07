from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "results/e4_control_transfer"

INPUT = OUT / "e4_test_userlevel.csv"

if not INPUT.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT}")

df = pd.read_csv(INPUT)

B = 5000
SEED = 2027

results = []

for control in sorted(df["control"].unique()):

    sub = df[df["control"] == control]

    base = (
        sub[sub["beta"] == 0.0]
        .set_index("user_id")
        .sort_index()
    )

    new = (
        sub[sub["beta"] == 1.0]
        .set_index("user_id")
        .sort_index()
    )

    if base.index.has_duplicates or new.index.has_duplicates:
        raise ValueError(f"Duplicate user IDs for {control}")

    if not base.index.equals(new.index):
        raise ValueError(f"User mismatch for {control}")

    users = base.index
    n = len(users)

    rng = np.random.default_rng(SEED)

    for metric in ["Recall@20", "NDCG@20"]:

        delta = (
            new[metric].to_numpy(dtype=float)
            - base[metric].to_numpy(dtype=float)
        )

        observed = float(delta.mean())

        bootstrap = np.empty(B, dtype=float)

        for b in range(B):
            indices = rng.integers(0, n, size=n)
            bootstrap[b] = delta[indices].mean()

        ci_low, ci_high = np.percentile(
            bootstrap, [2.5, 97.5]
        )

        results.append({
            "control": control,
            "metric": metric,
            "users": n,
            "baseline_mean": float(base[metric].mean()),
            "E4_mean": float(new[metric].mean()),
            "mean_difference": observed,
            "ci_lower_95": float(ci_low),
            "ci_upper_95": float(ci_high),
            "positive_users": int((delta > 1e-12).sum()),
            "negative_users": int((delta < -1e-12).sum()),
            "unchanged_users": int(
                (np.abs(delta) <= 1e-12).sum()
            ),
            "ci_excludes_zero": bool(
                ci_low > 0 or ci_high < 0
            )
        })

result = pd.DataFrame(results)

output = OUT / "e4_paired_bootstrap.csv"
result.to_csv(output, index=False)

print("\n=== E4 PAIRED BOOTSTRAP RESULTS ===")
print(result.round(6).to_string(index=False))

print("\nResults saved:", output)
print("Bootstrap samples:", B)
print("Random seed:", SEED)
