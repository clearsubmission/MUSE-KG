from pathlib import Path

root = Path("experiments/e5_mixed_relevance")

for mode in ["uniform", "tag"]:

    src = root / f"run_e5_{mode}_fine.py"
    dst = root / f"run_e7_{mode}_userlevel.py"

    s = src.read_text()

    # Initialize collection before the experiment loops.
    old = "records = []"
    new = "per_user_records = []\nrecords = []"

    assert s.count(old) == 1, (
        mode, "Could not locate records initialization"
    )
    s = s.replace(old, new, 1)

    # Record each user's outcomes after the
    # recommendation and relevance calculations.
    old = "            ndcgs.append(dcg / idcg)"

    new = """
            ndcgs.append(dcg / idcg)

            # E7: Record paired user-level outcomes.
            # The existing aggregate evaluator is unchanged.
            warm_hits_user = sum(
                int(i in warm_relevant)
                for i in recommended
            )
            cold_hits_user = sum(
                int(i in cold_relevant)
                for i in recommended
            )

            warm_recall_user = (
                warm_hits_user / len(warm_relevant)
                if warm_relevant else np.nan
            )
            cold_recall_user = (
                cold_hits_user / len(cold_relevant)
                if cold_relevant else np.nan
            )

            per_user_records.append({
                "user_id": int(u),
                "control": control,
                "method": "METHOD_NAME",
                "beta": float(beta),
                "warm_recall": warm_recall_user,
                "cold_recall": cold_recall_user,
                "overall_recall": float(
                    hits.sum() / len(relevant)
                ),
                "ndcg": float(dcg / idcg),
                "satisfaction": float(
                    indicator[top].mean()
                ),
                "n_warm_relevant": len(warm_relevant),
                "n_cold_relevant": len(cold_relevant),
                "warm_hits": warm_hits_user,
                "cold_hits": cold_hits_user,
            })
""".replace("METHOD_NAME", (
        "Uniform" if mode == "uniform"
        else "TagSpecific"
    ))

    assert s.count(old) == 1, (
        mode, "Could not locate NDCG insertion point"
    )
    s = s.replace(old, new.rstrip(), 1)

    # Write user-level data at the end of evaluation.
    old = "results = pd.DataFrame(records)"

    new = f"""
userlevel = pd.DataFrame(per_user_records)

userlevel.to_csv(
    OUT / "e7_{mode}_userlevel_validation.csv",
    index=False
)

print(
    "Saved user-level outcomes:",
    len(userlevel),
    flush=True
)

results = pd.DataFrame(records)
""".strip("\n")

    assert s.count(old) == 1
    s = s.replace(old, new, 1)

    # Do not overwrite previous aggregate results.
    old = f'OUT / "e5_{mode}_fine_validation.csv"'
    new = f'OUT / "e7_{mode}_aggregate_validation.csv"'

    assert s.count(old) == 1
    s = s.replace(old, new, 1)

    dst.write_text(s)
    print("Created:", dst)
