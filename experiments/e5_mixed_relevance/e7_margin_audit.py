from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("results/e5_mixed_relevance")
ROOT.mkdir(parents=True, exist_ok=True)

sources = {
    "Uniform": (
        "e5_uniform_fine_validation.csv",
        "e6_uniform_test.csv",
    ),
    "TagSpecific": (
        "e5_tag_fine_validation.csv",
        "e6_tag_test.csv",
    ),
}

tables = {
    method: {
        "val": pd.read_csv(ROOT / val_path),
        "test": pd.read_csv(ROOT / test_path),
    }
    for method, (val_path, test_path) in sources.items()
}

controls = sorted(
    tables["Uniform"]["val"]["control"].unique()
)

records = []

for target in [0.98, 0.95]:
    for margin in [0.0, 0.005, 0.01, 0.02, 0.03]:

        for control in controls:
            choices = []

            for method, datasets in tables.items():
                val = datasets["val"]
                test = datasets["test"]

                v = val[val.control == control]
                t = test[test.control == control]

                vbase = v[
                    np.isclose(v.beta, 0.0)
                ].iloc[0]

                tbase = t[
                    np.isclose(t.beta, 0.0)
                ].iloc[0]

                for _, candidate in v.iterrows():
                    strength = float(candidate.beta)

                    validation_retention = (
                        candidate["WarmRecall@20"]
                        / vbase["WarmRecall@20"]
                    )

                    # Zero intervention is always eligible
                    # as a baseline fallback.
                    is_baseline = np.isclose(
                        strength, 0.0
                    )

                    meets_margin = (
                        validation_retention
                        >= target + margin - 1e-9
                    )

                    if not (is_baseline or meets_margin):
                        continue

                    if (
                        candidate["Satisfaction@20"]
                        < 0.8 - 1e-9
                    ):
                        continue

                    matches = t[
                        np.isclose(t.beta, strength)
                    ]

                    assert len(matches) == 1, (
                        control, method, strength
                    )

                    measured = matches.iloc[0]

                    test_retention = (
                        measured["WarmRecall@20"]
                        / tbase["WarmRecall@20"]
                    )

                    choices.append({
                        "method": method,
                        "strength": strength,
                        "validation_retention":
                            validation_retention,
                        "validation_cold":
                            candidate["ColdRecall@20"],
                        "validation_ndcg":
                            candidate["NDCG@20"],
                        "test_retention":
                            test_retention,
                        "test_cold_gain": (
                            measured["ColdRecall@20"]
                            - tbase["ColdRecall@20"]
                        ),
                        "test_overall_gain": (
                            measured["Recall@20"]
                            - tbase["Recall@20"]
                        ),
                        "is_baseline": bool(is_baseline),
                    })

            if not choices:
                raise RuntimeError(
                    "Even baseline was unavailable for "
                    f"{control}, target={target}"
                )

            best = sorted(
                choices,
                key=lambda x: (
                    x["validation_cold"],
                    x["validation_ndcg"],
                    -x["strength"],
                ),
                reverse=True,
            )[0]

            records.append({
                "target": target,
                "margin": margin,
                "control": control,
                **best,
                "test_compliant": bool(
                    best["test_retention"]
                    >= target - 1e-9
                ),
            })

result = pd.DataFrame(records)

result.to_csv(
    ROOT / "e7_margin_sensitivity.csv",
    index=False,
)

summary = (
    result.groupby(["target", "margin"])
    .agg(
        compliant=("test_compliant", "sum"),
        baseline_fallbacks=("is_baseline", "sum"),
        mean_cold_gain=("test_cold_gain", "mean"),
        mean_overall_gain=("test_overall_gain", "mean"),
        worst_retention=("test_retention", "min"),
    )
    .reset_index()
)

summary.to_csv(
    ROOT / "e7_margin_summary.csv",
    index=False,
)

print("\n=== E7 SAFETY-MARGIN AUDIT ===")
print(summary.round(5).to_string(index=False))

print("\n=== BUDGET VIOLATIONS ===")
violations = result[~result.test_compliant]

if violations.empty:
    print("No violations in this retrospective audit.")
else:
    print(
        violations[
            [
                "target", "margin", "control",
                "method", "strength",
                "validation_retention",
                "test_retention",
            ]
        ].round(5).to_string(index=False)
    )

print("\nSaved:")
print(ROOT / "e7_margin_sensitivity.csv")
print(ROOT / "e7_margin_summary.csv")
