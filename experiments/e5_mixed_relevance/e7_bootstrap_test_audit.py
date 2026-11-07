from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path("results/e5_mixed_relevance")

tables = {
    "Uniform": pd.read_csv(
        ROOT / "e6_uniform_test.csv"
    ),
    "TagSpecific": pd.read_csv(
        ROOT / "e6_tag_test.csv"
    ),
}

baseline = tables["Uniform"][
    np.isclose(tables["Uniform"].beta, 0)
].set_index("control")

all_rows = []

for target in [98, 95]:
    policy = json.loads(
        (
            ROOT / f"e7_bootstrap_policy_{target}.json"
        ).read_text()
    )

    records = []

    for control, setting in sorted(policy.items()):
        method = setting["method"]
        gamma = float(setting["strength"])

        table = tables[method]

        selected = table[
            (table.control == control)
            & np.isclose(table.beta, gamma)
        ]

        assert len(selected) == 1, (
            control, method, gamma
        )

        row = selected.iloc[0]
        base = baseline.loc[control]

        retention = (
            row["WarmRecall@20"]
            / base["WarmRecall@20"]
        )

        records.append({
            "target": target,
            "control": control,
            "method": method,
            "gamma": gamma,
            "warm_retention": retention,
            "cold_gain": (
                row["ColdRecall@20"]
                - base["ColdRecall@20"]
            ),
            "overall_gain": (
                row["Recall@20"]
                - base["Recall@20"]
            ),
            "ndcg_gain": (
                row["NDCG@20"]
                - base["NDCG@20"]
            ),
            "satisfaction": row["Satisfaction@20"],
            "budget_satisfied": bool(
                retention >= target / 100 - 1e-9
            ),
        })

    result = pd.DataFrame(records)

    result.to_csv(
        ROOT / f"e7_bootstrap_policy_{target}_test.csv",
        index=False
    )

    all_rows.append(result)

    print(f"\n=== E7 {target}% BOOTSTRAP TEST AUDIT ===")
    print(result.round(5).to_string(index=False))

    print("\nBudget compliance:",
          int(result.budget_satisfied.sum()), "/ 8")
    print("Mean cold gain:",
          round(result.cold_gain.mean(), 5))
    print("Mean overall gain:",
          round(result.overall_gain.mean(), 5))
    print("Mean NDCG gain:",
          round(result.ndcg_gain.mean(), 5))

# Compare against existing E6 frozen policies.
summary = []

for target, bootstrap in zip([98, 95], all_rows):
    previous = pd.read_csv(
        ROOT / f"e6_policy_{target}_test_audit.csv"
    )

    summary.extend([
        {
            "target": target,
            "policy": "E6_PointEstimate",
            "compliant": int(
                previous.budget_satisfied.sum()
            ),
            "mean_cold_gain": (
                previous.cold_recall_gain.mean()
            ),
            "mean_overall_gain": (
                previous.overall_recall_gain.mean()
            ),
            "mean_ndcg_gain": previous.ndcg_gain.mean(),
        },
        {
            "target": target,
            "policy": "E7_Bootstrap",
            "compliant": int(
                bootstrap.budget_satisfied.sum()
            ),
            "mean_cold_gain": bootstrap.cold_gain.mean(),
            "mean_overall_gain": (
                bootstrap.overall_gain.mean()
            ),
            "mean_ndcg_gain": bootstrap.ndcg_gain.mean(),
        },
    ])

comparison = pd.DataFrame(summary)

comparison.to_csv(
    ROOT / "e7_vs_e6_policy_comparison.csv",
    index=False
)

print("\n=== E6 VERSUS E7 ===")
print(comparison.round(5).to_string(index=False))
