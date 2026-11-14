"""E8 selection: calibrate each controller on validation under the listener's budget, evaluate once on test.

Budget rule (as E7, normal-approximation lower bound): a setting is admissible for genre t and target r if
    mean_u[warm_u(setting) - r * warm_u(no bonus)] - z_{1-alpha} * sd / sqrt(n) >= 0   on validation users,
with alpha = 0.05 / (number of candidate settings x number of genres) (Bonferroni).
Systems: none | budget (gamma only, max cold recall) | creator (eta only, max tail hit) |
         cap (tag cap 4, no bonus, no selection) | joint (any gamma, eta, cap; max tail hit).
Tail/mid/head are popularity terciles of the new artists by held-out listeners in that split; they are used
only to score settings on validation and to report results on test, never by the deployed policy.
"""
import argparse, os, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm

p = argparse.ArgumentParser(); p.add_argument("--dir", default="results/e8"); p.add_argument("--targets", default="0.95,0.98")
a = p.parse_args()
ROOT = Path(os.environ.get("MUSE_ROOT", "/storage/bhull113/MUSE-KG")); D = ROOT / a.dir
TARGETS = [float(x) for x in a.targets.split(",")]
controls = sorted({Path(f).stem.split("_", 2)[2] for f in glob.glob(str(D / "e8_test_*.npz"))})
assert controls, "no test results found"

def metrics(z, ci):
    lis = z["listeners"]; nC = len(lis)
    terc = np.empty(nC, int); terc[np.argsort(lis, kind="stable")] = np.minimum(np.arange(nC) * 3 // nC, 2)
    ok = lis > 0
    hit = np.divide(z["hits_art"][ci], lis, out=np.zeros(nC), where=ok)
    ex = z["expo"][ci]; tot = max(ex.sum(), 1); nU = z["warm"].shape[1]
    out = {f"hit_{n}": hit[ok & (terc == t)].mean() for t, n in enumerate(["tail", "mid", "head"])}
    out.update({f"slots_{n}": ex[terc == t].sum() for t, n in enumerate(["tail", "mid", "head"])})
    out.update({f"share_{n}": ex[terc == t].sum() / tot for t, n in enumerate(["tail", "mid", "head"])})
    out.update(cold_slots=ex.sum(), tail_of_all_slots=ex[terc == 0].sum() / (nU * 20),
               warm=np.nanmean(z["warm"][ci]), cold=np.nanmean(z["coldr"][ci]), recall=z["recall"][ci].mean(),
               distinct_cold=int(z["distinct_cold"][ci]), distinct_total=int(z["distinct_total"][ci]))
    return out

SYSTEMS = {
    "none":    lambda g, e, c: g == 0 and e == 0 and c == 0,
    "budget":  lambda g, e, c: g > 0 and e == 0 and c == 0,
    "creator": lambda g, e, c: g == 0 and e > 0 and c == 0,
    "cap":     lambda g, e, c: g == 0 and e == 0 and c == 4,
    "joint":   lambda g, e, c: not (g == 0 and e == 0 and c == 0),
}
OBJ = {"budget": "cold", "creator": "hit_tail", "joint": "hit_tail"}
rows, picks = [], []
for ctrl in controls:
    zv = np.load(D / f"e8_validation_{ctrl}.npz"); zt = np.load(D / f"e8_test_{ctrl}.npz")
    combos = [tuple(x) for x in zv["combos"]]; assert combos == [tuple(x) for x in zt["combos"]]
    base = combos.index((0.0, 0.0, 0.0))
    Wv = zv["warm"]; users = ~np.isnan(Wv[base]); n = users.sum()
    mv = [metrics(zv, i) for i in range(len(combos))]; mt = [metrics(zt, i) for i in range(len(combos))]
    for r in TARGETS:
        for sysname, cond in SYSTEMS.items():
            cand = [i for i, (g, e, c) in enumerate(combos) if cond(g, e, c)]
            if sysname in ("none", "cap"):
                chosen = cand[0]; lcb = np.nan
            else:
                alpha = 0.05 / (len(cand) * len(controls)); z_ = norm.ppf(1 - alpha); best = None
                for i in cand:
                    s = Wv[i][users] - r * Wv[base][users]
                    lb = s.mean() - z_ * s.std(ddof=1) / np.sqrt(n)
                    if lb >= 0:
                        key = (mv[i][OBJ[sysname]], mv[i]["cold"])
                        if best is None or key > best[0]: best = (key, i, lb)
                chosen, lcb = (best[1], best[2]) if best else (base, np.nan)
            g, e, c = combos[chosen]; t = mt[chosen]; t0 = mt[base]
            rows.append(dict(target=r, system=sysname, control=ctrl, gamma=g, eta=e, cap=int(c), val_lcb=lcb,
                             fallback=(chosen == base and sysname not in ("none",)),
                             retention=t["warm"] / t0["warm"], compliant=t["warm"] / t0["warm"] >= r - 1e-12,
                             cold_gain=t["cold"] - t0["cold"],
                             **{k: t[k] for k in ["hit_tail", "hit_mid", "hit_head", "share_tail", "share_mid", "share_head",
                                                  "tail_of_all_slots", "distinct_cold", "distinct_total", "cold_slots",
                                                  "slots_tail", "slots_mid", "slots_head"]},
                             extra_tail=t["slots_tail"] - t0["slots_tail"], extra_mid=t["slots_mid"] - t0["slots_mid"],
                             extra_head=t["slots_head"] - t0["slots_head"]))
df = pd.DataFrame(rows); df.to_csv(D / "e8_selected_per_genre.csv", index=False)
agg = df.groupby(["target", "system"], sort=False).agg(
    compliant=("compliant", "sum"), fallbacks=("fallback", "sum"), retention=("retention", "mean"), worst=("retention", "min"),
    cold_gain=("cold_gain", "mean"), hit_tail=("hit_tail", "mean"), hit_mid=("hit_mid", "mean"), hit_head=("hit_head", "mean"),
    share_tail=("share_tail", "mean"), share_head=("share_head", "mean"), tail_of_all=("tail_of_all_slots", "mean"),
    distinct_cold=("distinct_cold", "mean"), extra_tail=("extra_tail", "sum"), extra_mid=("extra_mid", "sum"), extra_head=("extra_head", "sum"))
ext = agg[["extra_tail", "extra_mid", "extra_head"]]; tot = ext.sum(axis=1).replace(0, np.nan)
for k in ["tail", "mid", "head"]: agg[f"extra_share_{k}"] = agg[f"extra_{k}"] / tot
agg.to_csv(D / "e8_summary.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(agg.drop(columns=["extra_tail", "extra_mid", "extra_head"]).round(4).to_string())
print("\nPer-genre choices:"); print(df[["target", "system", "control", "gamma", "eta", "cap", "retention", "hit_tail"]].round(4).to_string(index=False))
