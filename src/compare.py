"""
Ablation table + charts across runs (run after evaluating every run):
    python -m src.compare --runs A_base B_pretrained C_clahe D_ordinal
"""
import os, json, argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=C.OUTPUT_DIR)
    args = ap.parse_args()

    rows, hist = {}, {}
    for name in args.runs:
        d = os.path.join(args.out, name)
        m = json.load(open(os.path.join(d, "metrics.json")))
        r = m["per_class"]["recall"]
        rows[name] = {"Params(M)": m["n_params"] / 1e6, "Accuracy": m["accuracy"],
                      "Precision": m["precision_macro"], "Recall": m["recall_macro"],
                      "F1": m["f1_macro"], "QWK": m["qwk"], "Minority recall": m["minority_recall"],
                      **{f"R_{c}": r[i] for i, c in enumerate(["NoDR", "Mild", "Mod", "Sev", "Prolif"])}}
        if "mc_dropout" in m:
            rows[name].update({"MC_Accuracy": m["mc_dropout"]["accuracy"], "MC_QWK": m["mc_dropout"]["qwk"],
                               "Unc_AUROC": m["mc_dropout"]["auroc_entropy_detects_error"]})
        hist[name] = pd.read_csv(os.path.join(d, "history.csv"))

    table = pd.DataFrame(rows).T
    print(table.round(4).to_string())
    table.round(4).to_csv(os.path.join(args.out, "ablation.csv"))

    main_cols = ["Accuracy", "Precision", "Recall", "F1", "QWK"]
    ax = table[main_cols].T.plot.bar(figsize=(11, 5), rot=0)
    lo = max(0, table[main_cols].values.min() - 0.1)
    ax.set_ylim(lo, 1); ax.set_title("Ablation: base LSD-HybridViT reproduction vs enhancements")
    ax.grid(axis="y", alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "ablation_metrics.png"), dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in hist.items():
        axes[0].plot(h.epoch, h.val_loss, label=name)
        axes[1].plot(h.epoch, h.val_qwk, label=name)
    axes[0].set_title("Validation loss"); axes[1].set_title("Validation QWK")
    for a in axes: a.set_xlabel("Epoch"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "training_curves.png"), dpi=150); plt.close()
    print(f"\nSaved ablation.csv, ablation_metrics.png, training_curves.png to {args.out}/")


if __name__ == "__main__":
    main()
