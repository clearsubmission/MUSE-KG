import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

ROOT = Path(".")
DATA = ROOT / "processed/lastfm_cold"
OUT = ROOT / "results/e4_control_transfer"
OUT.mkdir(parents=True, exist_ok=True)

K = 20
ANCHORS = 20
TARGET = 16
BETAS = [0.0, 1.0]

CONTROLS = {
    "rock": 73,
    "electronic": 18,
    "pop": 24,
    "indie": 81,
    "metal": 1,
    "jazz": 83,
    "ambient": 14,
    "acoustic": 185,
}

checkpoint = torch.load(
    ROOT / "results/e2_cold_start/mf_bpr_checkpoint.pt",
    map_location="cpu",
    weights_only=False
)

U = checkpoint["model_state"]["user_emb.weight"].numpy()
V = checkpoint["model_state"]["item_emb.weight"].numpy()

users = np.asarray(checkpoint["users"], dtype=int)
items = np.asarray(checkpoint["items"], dtype=int)

umap = {int(u): j for j, u in enumerate(users)}

train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "test.csv")

tags = pd.read_csv(
    DATA / "artist_tags_no_direct_leak.csv"
)[["item_id", "tag_id"]].drop_duplicates()

cold = np.sort(val.item_id.unique())
candidate_ids = np.concatenate([items, cold])

assert len(np.unique(candidate_ids)) == len(candidate_ids)

cmap = {int(i): j for j, i in enumerate(candidate_ids)}
imap = cmap.copy()

all_tags = np.sort(tags.tag_id.unique())
tmap = {int(t): j for j, t in enumerate(all_tags)}

tags = tags[tags.item_id.isin(imap)].copy()

A = csr_matrix(
    (
        np.ones(len(tags), dtype=np.float32),
        (
            tags.item_id.map(imap).to_numpy(dtype=int),
            tags.tag_id.map(tmap).to_numpy(dtype=int),
        ),
    ),
    shape=(len(candidate_ids), len(all_tags)),
)

A.data[:] = 1.0

# IDF is learned only from training artists.
known_rows = np.arange(len(items))

df = np.asarray(
    (A[known_rows] > 0).sum(axis=0)
).ravel()

idf = np.log((len(items) + 1) / (df + 1)) + 1.0

A_tfidf = normalize(
    A.multiply(idf).tocsr(), axis=1
)

known_tags = A_tfidf[known_rows]

cold_rows = np.arange(
    len(items), len(candidate_ids)
)

similarities = (
    A_tfidf[cold_rows] @ known_tags.T
).tocsr()

# Cache the same top-20 semantic anchors used in E3-D.
anchor_indices = []
anchor_sims = []

for r in range(len(cold)):
    row = similarities.getrow(r)

    if row.nnz == 0:
        anchor_indices.append(np.array([], dtype=int))
        anchor_sims.append(np.array([], dtype=float))
        continue

    selected = np.argsort(
        -row.data, kind="stable"
    )[:ANCHORS]

    anchor_indices.append(row.indices[selected])
    anchor_sims.append(row.data[selected])

# Build evaluation users and their histories.
seen = {}
for u, i in zip(train.user_id, train.item_id):
    seen.setdefault(int(u), set()).add(int(i))

targets = {}
for u, i in zip(val.user_id, val.item_id):
    if int(u) in umap:
        targets.setdefault(int(u), set()).add(int(i))

eval_users = sorted(targets)

U_eval = normalize(
    U[[umap[u] for u in eval_users]], axis=1
)

known_vecs = normalize(V, axis=1)
known_scores = U_eval @ known_vecs.T

# Training interactions are excluded from ranking.
for r, u in enumerate(eval_users):
    for item in seen.get(u, set()):
        j = cmap.get(item)
        if j is not None and j < len(items):
            known_scores[r, j] = -np.inf

discounts = 1.0 / np.log2(np.arange(2, K + 2))


def make_cold_vectors(control_tag, beta):
    result = []
    matching_anchor_weight = []

    tag_col = tmap[control_tag]

    # Permute control membership across known artists only.
    # The genuine candidate control labels remain unchanged.
    real_anchor_labels = (
        A[known_rows, tag_col]
        .toarray()
        .ravel()
        .astype(np.float32)
    )

    shuffle_seed = int(
        os.environ.get("MUSE_SHUFFLE_SEED", "2027")
    )

    rng = np.random.default_rng(
        shuffle_seed + int(control_tag)
    )

    shuffled_anchor_labels = rng.permutation(
        real_anchor_labels
    )

    for idx, sims in zip(anchor_indices, anchor_sims):
        if len(idx) == 0:
            result.append(np.zeros(V.shape[1], dtype=np.float32))
            matching_anchor_weight.append(0.0)
            continue

        matches = shuffled_anchor_labels[idx]

        logits = sims + beta * matches
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()

        result.append(
            (V[idx] * weights[:, None]).sum(axis=0)
        )

        matching_anchor_weight.append(
            float((weights * matches).sum())
        )

    return (
        normalize(np.asarray(result, dtype=np.float32), axis=1),
        float(np.mean(matching_anchor_weight)),
    )


def best(indices, scores, n):
    order = np.lexsort(
        (candidate_ids[indices], -scores[indices])
    )
    return indices[order[:n]]


def recommend_constrained(scores, indicator):
    finite = np.flatnonzero(np.isfinite(scores))

    top = best(finite, scores, K)

    if indicator[top].sum() >= TARGET:
        return top

    eligible = finite[indicator[finite] > 0]
    other = finite[indicator[finite] == 0]

    if len(eligible) < TARGET:
        return None

    selected = np.concatenate([
        best(eligible, scores, TARGET),
        best(other, scores, K - TARGET),
    ])

    return best(selected, scores, K)


records = []

for control, tag_id in CONTROLS.items():
    print(f"\nControl: {control}", flush=True)

    if tag_id not in tmap:
        raise ValueError(f"Tag missing: {control}")

    indicator = (
        A[:, tmap[tag_id]].toarray().ravel() > 0
    ).astype(np.int8)

    for beta in BETAS:
        cold_vecs, anchor_match_weight = make_cold_vectors(
            tag_id, beta
        )

        cold_scores = U_eval @ cold_vecs.T
        scores_all = np.concatenate(
            [known_scores, cold_scores], axis=1
        )

        recalls = []
        ndcgs = []
        conditional_recalls = []
        conditional_ndcgs = []
        satisfaction = []
        recovered = set()
        feasible = 0

        for r, u in enumerate(eval_users):
            scores = scores_all[r]

            top = recommend_constrained(
                scores, indicator
            )

            if top is None:
                raise RuntimeError(
                    f"Constraint infeasible for {control}, user {u}"
                )

            feasible += 1
            recommended = candidate_ids[top]
            relevant = targets[u]

            hits = np.array(
                [int(i in relevant) for i in recommended],
                dtype=np.float32
            )

            recalls.append(
                float(hits.sum() / len(relevant))
            )

            dcg = float((hits * discounts).sum())
            idcg = float(
                discounts[:min(K, len(relevant))].sum()
            )
            ndcgs.append(dcg / idcg)

            satisfaction.append(
                float(indicator[top].mean())
            )

            recovered.update(
                int(i) for i in recommended
                if i in relevant
            )

            matching_relevant = {
                i for i in relevant
                if indicator[cmap[i]] > 0
            }

            if matching_relevant:
                chits = np.array(
                    [
                        int(i in matching_relevant)
                        for i in recommended
                    ],
                    dtype=np.float32
                )

                conditional_recalls.append(
                    float(chits.sum() / len(matching_relevant))
                )

                conditional_dcg = float(
                    (chits * discounts).sum()
                )

                conditional_idcg = float(
                    discounts[
                        :min(K, len(matching_relevant))
                    ].sum()
                )

                conditional_ndcgs.append(
                    conditional_dcg / conditional_idcg
                )

        row = {
            "control": control,
            "beta": beta,
            "Recall@20": float(np.mean(recalls)),
            "NDCG@20": float(np.mean(ndcgs)),
            "ConditionalRecall@20": float(
                np.mean(conditional_recalls)
            ) if conditional_recalls else np.nan,
            "ConditionalNDCG@20": float(
                np.mean(conditional_ndcgs)
            ) if conditional_ndcgs else np.nan,
            "Satisfaction@20": float(
                np.mean(satisfaction)
            ),
            "RecoveredColdArtists": len(recovered),
            "MeanControlAnchorWeight": anchor_match_weight,
            "FeasibleUsers": feasible,
        }

        records.append(row)

        print(
            f"  beta={beta:.1f} "
            f"recall={row['Recall@20']:.5f} "
            f"ndcg={row['NDCG@20']:.5f} "
            f"conditional={row['ConditionalRecall@20']:.5f} "
            f"anchor_weight={anchor_match_weight:.3f}",
            flush=True
        )

results = pd.DataFrame(records)

results.to_csv(
    OUT / (
        f"e4_shuffled_seed"
        f"{os.environ.get('MUSE_SHUFFLE_SEED', '2027')}.csv"
    ),
    index=False
)

print("\n=== E4 RESULTS ===")
print(results.round(5).to_string(index=False))


print("\nTest evaluation completed.")
print("Beta=0 is the test control-independent baseline.")
print("Beta=1 uses the validation-selected global setting.")
