from pathlib import Path

src = Path("experiments/e4_control_transfer/run_e4.py")
dst = Path("experiments/e5_mixed_relevance/run_e5_validation.py")

s = src.read_text()

changes = [
    (
        'DATA = ROOT / "processed/lastfm_cold"',
        'DATA = ROOT / "processed/e5_mixed_relevance"'
    ),
    (
        'OUT = ROOT / "results/e4_control_transfer"',
        'OUT = ROOT / "results/e5_mixed_relevance"'
    ),
    (
        'BETAS = [0.0, 0.5, 1.0, 2.0]',
        'BETAS = [0.0, 1.0]'
    ),
    (
        'ROOT / "results/e2_cold_start/mf_bpr_checkpoint.pt"',
        'ROOT / "results/e5_mixed_relevance/e5_bpr_checkpoint.pt"'
    ),
    (
        'DATA / "artist_tags_no_direct_leak.csv"',
        'DATA / "tags_without_direct_heldout_pairs.csv"'
    ),
    (
        'cold = np.sort(val.item_id.unique())',
        '''cold = np.sort(
    pd.read_csv(DATA / "cold_validation.csv")
    .item_id.unique()
)'''
    ),
    (
        'OUT / "e4_validation_results.csv"',
        'OUT / "e5_validation_results.csv"'
    ),
]

for old, new in changes:
    count = s.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one match, found {count}: {old}"
        )
    s = s.replace(old, new, 1)

# Track category-specific results for each evaluated user.
old = '''        recalls = []
        ndcgs = []
        conditional_recalls = []'''

new = '''        recalls = []
        ndcgs = []
        warm_recalls = []
        cold_recalls = []
        warm_ndcgs = []
        cold_ndcgs = []
        conditional_recalls = []'''

assert s.count(old) == 1
s = s.replace(old, new, 1)

old = '''            relevant = targets[u]

            hits = np.array('''

new = '''            relevant = targets[u]

            warm_relevant = {
                i for i in relevant
                if i in known_item_set
            }
            cold_relevant = {
                i for i in relevant
                if i in cold_item_set
            }

            hits = np.array('''

assert s.count(old) == 1
s = s.replace(old, new, 1)

old = '''            satisfaction.append(
                float(indicator[top].mean())
            )'''

new = '''            for relevant_group, group_recall, group_ndcg in [
                (warm_relevant, warm_recalls, warm_ndcgs),
                (cold_relevant, cold_recalls, cold_ndcgs),
            ]:
                if relevant_group:
                    group_hits = np.array(
                        [
                            int(i in relevant_group)
                            for i in recommended
                        ],
                        dtype=np.float32
                    )
                    group_recall.append(
                        float(
                            group_hits.sum()
                            / len(relevant_group)
                        )
                    )

                    group_dcg = float(
                        (group_hits * discounts).sum()
                    )
                    group_idcg = float(
                        discounts[
                            :min(K, len(relevant_group))
                        ].sum()
                    )
                    group_ndcg.append(
                        group_dcg / group_idcg
                    )

            satisfaction.append(
                float(indicator[top].mean())
            )'''

assert s.count(old) == 1
s = s.replace(old, new, 1)

old = '''            recovered.update(
                int(i) for i in recommended
                if i in relevant
            )'''

new = '''            recovered.update(
                int(i) for i in recommended
                if i in cold_relevant
            )'''

assert s.count(old) == 1
s = s.replace(old, new, 1)

old = '''            "NDCG@20": float(np.mean(ndcgs)),
            "ConditionalRecall@20":'''

new = '''            "NDCG@20": float(np.mean(ndcgs)),
            "WarmRecall@20": float(np.mean(warm_recalls)),
            "ColdRecall@20": float(np.mean(cold_recalls)),
            "WarmNDCG@20": float(np.mean(warm_ndcgs)),
            "ColdNDCG@20": float(np.mean(cold_ndcgs)),
            "WarmUsers": len(warm_recalls),
            "ColdUsers": len(cold_recalls),
            "ConditionalRecall@20":'''

assert s.count(old) == 1
s = s.replace(old, new, 1)

# Insert reference sets and disjointness checks
# before evaluation begins.
marker = 'records = []\n\nfor control, tag_id in CONTROLS.items():'

insert = '''known_item_set = set(map(int, items))
cold_item_set = set(map(int, cold))

assert known_item_set.isdisjoint(cold_item_set)

warm_validation = pd.read_csv(
    DATA / "warm_validation.csv"
)
cold_validation = pd.read_csv(
    DATA / "cold_validation.csv"
)

assert set(warm_validation.item_id).issubset(
    known_item_set
)
assert set(cold_validation.item_id).issubset(
    cold_item_set
)

assert len(warm_validation) + len(cold_validation) == len(val)

assert set(val.item_id).issubset(
    known_item_set | cold_item_set
)

print(
    "E5 candidate catalog:",
    len(known_item_set), "known +",
    len(cold_item_set), "cold artists",
    flush=True
)

records = []

for control, tag_id in CONTROLS.items():'''

assert s.count(marker) == 1
s = s.replace(marker, insert, 1)

# E3-D reference is from a different split.
# Remove the old baseline comparison.
marker = "# Audit whether beta=0 reproduces E3-D."

assert s.count(marker) == 1
s = s.split(marker)[0]

s += '''
print("\\nE5 mixed-relevance validation completed.")
'''

dst.write_text(s)
print("Created:", dst)
