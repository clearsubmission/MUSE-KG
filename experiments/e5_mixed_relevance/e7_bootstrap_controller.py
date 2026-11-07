from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path("results/e5_mixed_relevance")
ROOT.mkdir(parents=True, exist_ok=True)

SEED = 2027
BOOTSTRAPS = 2000
ALPHA = 0.05
TARGETS = [0.98, 0.95]

rng = np.random.default_rng(SEED)

sources = {
    "Uniform": (
        "e7_uniform_userlevel_validation.csv",
        "e5_uniform_fine_validation.csv",
    ),
    "TagSpecific": (
        "e7_tag_userlevel_validation.csv",
        "e5_tag_fine_validation.csv",
    ),
}

user_frames = []
aggregate_frames = []

for method, (user_file, aggregate_file) in sources.items():
    users = pd.read_csv(ROOT / user_file)
    aggregate = pd.read_csv(ROOT / aggregate_file)

    users["method"] = method
    aggregate["method"] = method

    user_frames.append(users)
    aggregate_frames.append(aggregate)

users = pd.concat(user_frames, ignore_index=True)
aggregate = pd.concat(aggregate_frames, ignore_index=True)

controls = sorted(users.control.unique())

# Two methods × eight nonzero strengths × eight controls.
N_CANDIDATES = 2 * 8 * len(controls)

# Conservative adjustment across the full
# set of candidate interventions.
adjusted_alpha = ALPHA / N_CANDIDATES

records = []
policies = {}

for target in TARGETS:
    policy = {}

    for control in controls:
        control_data = users[
            users.control == control
        ]

        # Construct a single paired user-indexed matrix,
        # ensuring all candidate outcomes correspond
        # to the same evaluation users.
        baseline = (
            control_data[
                (control_data.method == "Uniform")
                & np.isclose(control_data.beta, 0.0)
            ][["user_id", "warm_recall"]]
            .dropna()
            .set_index("user_id")
            .sort_index()
        )

        user_ids = baseline.index
        baseline_values = baseline["warm_recall"].to_numpy()

        n_users = len(user_ids)
        assert n_users > 0

        # Same bootstrap user samples for each candidate
        # within a musical control.
        bootstrap_indices = rng.integers(
            0, n_users,
            size=(BOOTSTRAPS, n_users),
        )

        candidates = []

        for method in sources:
            method_data = control_data[
                control_data.method == method
            ]

            for strength in sorted(method_data.beta.unique()):
                if np.isclose(strength, 0.0):
                    continue

                treated = (
                    method_data[
                        np.isclose(method_data.beta, strength)
                    ][["user_id", "warm_recall"]]
                    .set_index("user_id")
                    .reindex(user_ids)
                )

                treated_values = treated[
                    "warm_recall"
                ].to_numpy()

                assert np.isfinite(treated_values).all()

                slack = (
                    treated_values
                    - target * baseline_values
                )

                bootstrap_slack = (
                    slack[bootstrap_indices].mean(axis=1)
                )

                lcb = float(
                    np.quantile(
                        bootstrap_slack,
                        adjusted_alpha,
                    )
                )

                point_slack = float(slack.mean())

                matching = aggregate[
                    (aggregate.control == control)
                    & (aggregate.method == method)
                    & np.isclose(aggregate.beta, strength)
                ]

                assert len(matching) == 1
                metric = matching.iloc[0]

                candidate = {
                    "control": control,
                    "target": target,
                    "method": method,
                    "strength": float(strength),
                    "mean_slack": point_slack,
                    "lower_slack_bound": lcb,
                    "validation_cold_recall": float(
                        metric["ColdRecall@20"]
                    ),
                    "validation_warm_recall": float(
                        metric["WarmRecall@20"]
                    ),
                    "validation_ndcg": float(
                        metric["NDCG@20"]
                    ),
                    "satisfaction": float(
                        metric["Satisfaction@20"]
                    ),
                    "eligible": bool(
                        lcb >= 0.0
                        and metric["Satisfaction@20"] >= 0.8 - 1e-9
                    ),
                }

                candidates.append(candidate)
                records.append(candidate)

        eligible = [
            candidate
            for candidate in candidates
            if candidate["eligible"]
        ]

        if eligible:
            best = max(
                eligible,
                key=lambda x: (
                    x["validation_cold_recall"],
                    x["validation_ndcg"],
                    -x["strength"],
                ),
            )

            policy[control] = {
                "method": best["method"],
                "strength": best["strength"],
                "lower_slack_bound": best["lower_slack_bound"],
                "validation_cold_recall":
                    best["validation_cold_recall"],
            }
        else:
            # Baseline fallback: no intervention.
            policy[control] = {
                "method": "Uniform",
                "strength": 0.0,
                "lower_slack_bound": None,
                "validation_cold_recall": float(
                    aggregate[
                        (aggregate.control == control)
                        & (aggregate.method == "Uniform")
                        & np.isclose(aggregate.beta, 0.0)
                    ]["ColdRecall@20"].iloc[0]
                ),
            }

    policies[target] = policy

    output = (
        ROOT
        / f"e7_bootstrap_policy_{int(target * 100)}.json"
    )

    output.write_text(json.dumps(policy, indent=2))

    print(f"\n=== E7 BOOTSTRAP POLICY {target:.0%} ===")

    for control, setting in sorted(policy.items()):
        print(
            f"{control:12s} "
            f"{setting['method']:12s} "
            f"gamma={setting['strength']:.3f} "
            f"LCB={setting['lower_slack_bound']}"
        )

    print("Saved:", output)

audit = pd.DataFrame(records)

audit.to_csv(
    ROOT / "e7_bootstrap_candidate_audit.csv",
    index=False,
)

print("\n=== ELIGIBLE CANDIDATES BY TARGET ===")
print(
    audit.groupby("target")["eligible"]
    .sum()
    .to_string()
)

print("\nBootstrap controller completed.")
