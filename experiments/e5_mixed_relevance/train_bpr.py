from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(".")
DATA = ROOT / "processed/e5_mixed_relevance"
OUT = ROOT / "results/e5_mixed_relevance"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
DIM = 64
EPOCHS = 30
BATCH = 1024
LR = 0.002
REG = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Keep CPU execution reproducible and modest
torch.set_num_threads(4)
device = torch.device("cpu")

train = pd.read_csv(DATA / "train.csv")
assert not train.empty
assert not train[["user_id", "item_id"]].duplicated().any()

users = np.sort(train.user_id.unique())
items = np.sort(train.item_id.unique())

umap = {int(u): j for j, u in enumerate(users)}
imap = {int(i): j for j, i in enumerate(items)}

u_idx = train.user_id.map(umap).to_numpy(dtype=np.int64)
i_idx = train.item_id.map(imap).to_numpy(dtype=np.int64)

user_positive = [
    set() for _ in range(len(users))
]

for u, i in zip(u_idx, i_idx):
    user_positive[int(u)].add(int(i))

rng = np.random.default_rng(SEED)


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)

        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)

    def forward(self, u, pos, neg):
        ue = self.user_emb(u)
        pe = self.item_emb(pos)
        ne = self.item_emb(neg)

        pos_score = (ue * pe).sum(dim=1)
        neg_score = (ue * ne).sum(dim=1)

        loss = torch.nn.functional.softplus(
            neg_score - pos_score
        ).mean()

        reg = (
            ue.square().sum(dim=1)
            + pe.square().sum(dim=1)
            + ne.square().sum(dim=1)
        ).mean()

        return loss + REG * reg


def sample_negatives(batch_users):
    negatives = rng.integers(
        0, len(items), size=len(batch_users),
        dtype=np.int64
    )

    for r, user in enumerate(batch_users):
        positives = user_positive[int(user)]

        while int(negatives[r]) in positives:
            negatives[r] = rng.integers(len(items))

    return negatives


model = BPRMF(
    len(users), len(items), DIM
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=LR
)

print("=== E5 BPR TRAINING ===", flush=True)
print("Users:", len(users), flush=True)
print("Known artists:", len(items), flush=True)
print("Training pairs:", len(train), flush=True)
print("Device:", device, flush=True)

for epoch in range(1, EPOCHS + 1):
    model.train()

    permutation = rng.permutation(len(train))
    losses = []

    for start in range(0, len(train), BATCH):
        batch = permutation[start:start + BATCH]

        ub_np = u_idx[batch]
        pb_np = i_idx[batch]
        nb_np = sample_negatives(ub_np)

        ub = torch.from_numpy(ub_np).to(device)
        pb = torch.from_numpy(pb_np).to(device)
        nb = torch.from_numpy(nb_np).to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model(ub, pb, nb)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach()))

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"loss={np.mean(losses):.6f}",
            flush=True
        )

checkpoint = {
    "model_state": {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    },
    "users": users,
    "items": items,
    "seed": SEED,
    "embedding_dim": DIM,
    "epochs": EPOCHS,
    "train_file": str(DATA / "train.csv"),
}

path = OUT / "e5_bpr_checkpoint.pt"
torch.save(checkpoint, path)

print("\nSaved:", path)

# Check that no test or validation cold artist
# appears in the collaborative training catalog.
for filename in ["cold_validation.csv", "cold_test.csv"]:
    split = pd.read_csv(DATA / filename)
    overlap = np.intersect1d(items, split.item_id.unique())
    assert len(overlap) == 0, (
        f"Cold artist leakage from {filename}"
    )

print("Cold-artist separation verified.")
print("Training completed.")
