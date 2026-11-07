from pathlib import Path

root = Path("experiments/e5_mixed_relevance")
src = root / "run_e5_validation.py"

original = src.read_text()

for mode in ["uniform", "tag"]:
    s = original

    old = "BETAS = [0.0, 1.0]"
    new = "BETAS = [0.0, 0.05, 0.1, 0.2, 0.4]"
    assert s.count(old) == 1
    s = s.replace(old, new, 1)

    old = """        cold_vecs, anchor_match_weight = make_cold_vectors(
            tag_id, beta
        )

        cold_scores = U_eval @ cold_vecs.T"""

    if mode == "uniform":
        bonus = "np.ones(len(cold), dtype=np.float32)"
    else:
        bonus = "indicator[len(items):].astype(np.float32)"

    new = f"""        # Keep transferred representations unchanged.
        cold_vecs, anchor_match_weight = make_cold_vectors(
            tag_id, 0.0
        )

        cold_scores = U_eval @ cold_vecs.T

        bonus_mask = {bonus}
        cold_scores = (
            cold_scores + beta * bonus_mask[None, :]
        )"""

    assert s.count(old) == 1
    s = s.replace(old, new, 1)

    old = 'OUT / "e5_validation_results.csv"'
    new = f'OUT / "e5_{mode}_bonus_validation.csv"'

    assert s.count(old) == 1
    s = s.replace(old, new, 1)

    dst = root / f"run_e5_{mode}_bonus.py"
    dst.write_text(s)

    print("Created:", dst)
