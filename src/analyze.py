"""Compare singles / equal vote / weighted vote / stacking, with and without temperature scaling,
on in-domain (Test-A) and external (Test-B) data. Everything is fit on the in-domain validation split only.
Usage: python -m src.analyze --tags std robust --seeds 0 1 2
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from .common import CLASSES, MODELS, OUT
from . import metrics as M
from . import ensemble as E

DOMAINS = ["testA", "testB"]
ENS = ["equal_vote", "weighted_vote", "stacking"]


def load(tag):
    z = np.load(f"{OUT}/logits/{tag}.npz")
    return {k: z[k] for k in z.files}


def run(tag, seed, models):
    Z = {m: load(f"{m}_{tag}_s{seed}") for m in models}
    splits = ["val"] + DOMAINS
    y = {s: Z[models[0]][f"{s}_y"] for s in splits}
    L = {m: {s: Z[m][f"{s}_logits"] for s in splits} for m in models}
    P = {m: {s: M.softmax(L[m][s]) for s in splits} for m in models}
    members = lambda s: [P[m][s] for m in models]

    w = E.search_weights(np.array(members("val")), y["val"])
    clf, val_oof = E.fit_stacker(members("val"), y["val"])

    ens_p = {"equal_vote": {s: E.equal_vote(members(s)) for s in splits},
             "weighted_vote": {s: E.weighted_vote(np.array(members(s)), w) for s in splits},
             "stacking": {s: E.stack_predict(clf, members(s)) for s in splits}}
    ens_p["stacking"]["val"] = val_oof  # out-of-fold, so val temperature is not fit on in-sample predictions

    R = {m: {s: (P[m][s], L[m][s]) for s in splits} for m in models}
    for k, d in ens_p.items():
        R[k] = {s: (p, np.log(p + 1e-8)) for s, p in d.items()}

    rows, probs = [], {}
    for name, d in R.items():
        T = M.fit_temperature(d["val"][1], y["val"])
        for variant, T_ in [("", 1.0), ("+TS", T)]:
            for s in DOMAINS:
                p = M.softmax(d[s][1] / T_) if variant else d[s][0]
                pred = p.argmax(1); rec = M.per_class_recall(y[s], pred)
                rows.append({"tag": tag, "seed": seed, "method": name + variant, "domain": s,
                             "acc": float((pred == y[s]).mean()), "macro_f1": M.macro_f1(y[s], pred),
                             "ece": M.ece(p, y[s]), "nll": M.nll(p, y[s]), "T": T_,
                             **{f"recall_{c}": r for c, r in zip(CLASSES, rec)}})
                probs[(name + variant, s)] = p
    return rows, {"weights": dict(zip(models, map(float, w))), "y": y, "P": P, "probs": probs}


def selective_curves(info, models, tag, best="weighted_vote+TS"):
    """Accuracy-vs-coverage when abstaining on the most uncertain cases."""
    recs = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, s in zip(ax, DOMAINS):
        p, y = info["probs"][(best, s)], info["y"][s]
        mem = [info["P"][m][s] for m in models]
        scores = {"1-max prob": 1 - p.max(1), "entropy": M.entropy(p), "model disagreement (MI)": M.mutual_info(mem)}
        corr = p.argmax(1) == y
        for nm, sc in scores.items():
            order = np.argsort(sc); n = np.arange(1, len(y) + 1)
            acc = np.cumsum(corr[order]) / n
            a.plot(n / len(y), acc, label=nm)
            for c in (1.0, 0.9, 0.8, 0.7):
                recs.append({"tag": tag, "domain": s, "score": nm, "coverage": c,
                             "accuracy": float(acc[int(c * len(y)) - 1])})
        a.set_title(f"{s} ({best})"); a.set_xlabel("coverage"); a.grid(alpha=.3)
    ax[0].set_ylabel("accuracy on retained"); ax[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/results/coverage_{tag}.png", dpi=150); plt.close()
    return recs


def reliability(info, tag, methods=("weighted_vote", "weighted_vote+TS")):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, s in zip(ax, DOMAINS):
        for m in methods:
            p, y = info["probs"][(m, s)], info["y"][s]
            conf, ok = p.max(1), p.argmax(1) == y
            edges = np.linspace(0, 1, 11); xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                k = (conf > lo) & (conf <= hi)
                if k.sum() >= 5:
                    xs.append(conf[k].mean()); ys.append(ok[k].mean())
            a.plot(xs, ys, "o-", label=f"{m} (ECE {M.ece(p, y):.3f})")
        a.plot([0, 1], [0, 1], "k--"); a.set_title(f"Reliability - {s}"); a.set_xlabel("confidence")
        a.legend(fontsize=7)
    ax[0].set_ylabel("accuracy"); plt.tight_layout()
    plt.savefig(f"{OUT}/results/reliability_{tag}.png", dpi=150); plt.close()


def confusion_figure(info, tag, method="weighted_vote"):
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(10, 4))
    for axis, domain in zip(np.atleast_1d(axes), DOMAINS):
        y = info["y"][domain]
        pred = info["probs"][(method, domain)].argmax(1)
        cm = M.confusion(y, pred, len(CLASSES))
        axis.imshow(cm, cmap="Blues")
        axis.set_title(domain)
        axis.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
        axis.set_yticks(range(len(CLASSES)), CLASSES)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                axis.text(j, i, cm[i, j], ha="center", va="center")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    fig.suptitle(f"{tag} {method} confusion matrices")
    fig.tight_layout()
    fig.savefig(f"{OUT}/results/confusion_{tag}_{method}.png", dpi=180)
    plt.close(fig)


def stats(info, models, tag, best_single):
    out = []
    for s in DOMAINS:
        y = info["y"][s]
        for name in models + ENS:
            pred = info["probs"][(name, s)].argmax(1); lo, hi = M.bootstrap_ci(y, pred)
            out.append({"tag": tag, "domain": s, "method": name, "macro_f1": M.macro_f1(y, pred),
                        "ci_lo": lo, "ci_hi": hi})
        a = info["probs"][(best_single, s)].argmax(1) == y
        for ens in ENS:
            b = info["probs"][(ens, s)].argmax(1) == y
            n01, n10 = int((a & ~b).sum()), int((~a & b).sum())  # McNemar discordant pairs (exact test)
            pv = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            out.append({"tag": tag, "domain": s, "method": f"McNemar {ens} vs {best_single}",
                        "single_only_correct": n01, "ens_only_correct": n10, "p_value": pv})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["std"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--models", nargs="+", default=MODELS)
    a = ap.parse_args()
    os.makedirs(f"{OUT}/results", exist_ok=True)

    all_rows, unc, st = [], [], []
    per_class, confusion_rows = [], []
    for tag in a.tags:
        for seed in a.seeds:
            rows, info = run(tag, seed, a.models); all_rows += rows
            print(tag, seed, "ensemble weights", info["weights"])
            if seed == a.seeds[0]:
                unc += selective_curves(info, a.models, tag); reliability(info, tag)
                confusion_figure(info, tag)
                # best single model chosen on validation macro-F1 (not on test)
                vf1 = {m: M.macro_f1(info["y"]["val"], info["P"][m]["val"].argmax(1)) for m in a.models}
                st += stats(info, a.models, tag, max(vf1, key=vf1.get))
            for method, domain in [(m, d) for m in a.models + ENS for d in DOMAINS]:
                if (method, domain) not in info["probs"]:
                    continue
                y = info["y"][domain]
                pred = info["probs"][(method, domain)].argmax(1)
                cm = M.confusion(y, pred, len(CLASSES))
                recalls = M.per_class_recall(y, pred)
                for class_index, class_name in enumerate(CLASSES):
                    per_class.append({
                        "tag": tag, "seed": seed, "method": method, "domain": domain,
                        "class": class_name, "recall": float(recalls[class_index]),
                        "support": int(cm[class_index].sum()),
                    })
                for true_index, true_name in enumerate(CLASSES):
                    for pred_index, pred_name in enumerate(CLASSES):
                        confusion_rows.append({
                            "tag": tag, "seed": seed, "method": method, "domain": domain,
                            "true_class": true_name, "predicted_class": pred_name,
                            "count": int(cm[true_index, pred_index]),
                        })
    df = pd.DataFrame(all_rows); df.to_csv(f"{OUT}/results/all_runs.csv", index=False)
    summ = df.groupby(["tag", "method", "domain"])[["acc", "macro_f1", "ece", "nll"]].agg(["mean", "std"]).round(4)
    summ.to_csv(f"{OUT}/results/summary.csv")
    pd.DataFrame(unc).to_csv(f"{OUT}/results/selective_accuracy.csv", index=False)
    pd.DataFrame(st).to_csv(f"{OUT}/results/ci_and_mcnemar.csv", index=False)
    gap = df.pivot_table(index=["tag", "method"], columns="domain", values="macro_f1", aggfunc="mean")
    gap["domain_gap(A-B)"] = gap["testA"] - gap["testB"]
    gap.round(4).to_csv(f"{OUT}/results/domain_gap.csv")
    pd.DataFrame(per_class).to_csv(f"{OUT}/results/per_class_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(f"{OUT}/results/confusion_matrices.csv", index=False)
    publication = df.groupby(["tag", "method", "domain"], as_index=False).agg(
        accuracy_mean=("acc", "mean"), accuracy_std=("acc", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        ece_mean=("ece", "mean"), ece_std=("ece", "std"),
        nll_mean=("nll", "mean"), nll_std=("nll", "std"),
    ).round(4)
    publication.to_csv(f"{OUT}/results/publication_summary.csv", index=False)
    print(summ.to_string()); print(gap.round(4).to_string())


if __name__ == "__main__":
    main()
