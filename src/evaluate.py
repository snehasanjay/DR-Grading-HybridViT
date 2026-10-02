"""
Evaluate a trained run on the held-out TEST split (never used during training).

    python -m src.evaluate --run D_ordinal                  # standard (flip TTA)
    python -m src.evaluate --run D_ordinal --mc-samples 30  # + MC-dropout uncertainty

Outputs in outputs/<run>/:
  metrics.json, classification_report.txt, confusion_matrix.png, test_predictions.csv
  with --mc-samples: uncertainty.json, referral_curve.png
"""
import os, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, cohen_kappa_score

from . import config as C
from .dataset import DRDataset, get_transforms
from .models import build_model, enable_mc_dropout
from .utils import get_device, compute_metrics, save_json


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt["cfg"]); cfg["pretrained"] = False      # weights come from the checkpoint
    model = build_model(**cfg)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def aggregate(outs, head):
    """Average several raw outputs (list of (N, .) tensors) -> probs, pred, expected grade."""
    if head == "ce":
        p = torch.stack([o.softmax(1) for o in outs]).mean(0)
        pred = p.argmax(1)
        grade = (p * torch.arange(p.shape[1], dtype=p.dtype)).sum(1)
        return p, pred, grade
    s = torch.stack([torch.sigmoid(o) for o in outs]).mean(0)          # P(y > k)
    cum = torch.cat([torch.ones_like(s[:, :1]), s, torch.zeros_like(s[:, :1])], 1)
    p = (cum[:, :-1] - cum[:, 1:]).clamp(min=1e-8); p = p / p.sum(1, keepdim=True)
    return p, (s > 0.5).sum(1), s.sum(1)


@torch.no_grad()
def collect(model, loader, device, n_passes, tta):
    """Returns list over passes of raw outputs for the whole loader, and targets."""
    per_pass = [[] for _ in range(n_passes)]; ys = []
    for x, y in loader:
        x = x.to(device)
        for t in range(n_passes):
            if tta:
                views = [x, x.flip(3), x.flip(2)]
                per_pass[t].append(torch.stack([model(v).float().cpu() for v in views]))
            else:
                per_pass[t].append(model(x).float().cpu().unsqueeze(0))
        ys.append(y)
    # each pass -> (views, N, out_dim)
    return [torch.cat(p, 1) for p in per_pass], torch.cat(ys).numpy()


def plot_confusion(cm, path, title):
    cm = np.array(cm)
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    annot = np.array([[f"{cm[i,j]}\n{cm_norm[i,j]:.0%}" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1,
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def uncertainty_analysis(y, pred, probs, grade_passes, run_dir, run):
    """MC-dropout uncertainty: does high uncertainty flag the model's mistakes, and how much
    do accuracy/QWK improve if the most uncertain cases are referred to an ophthalmologist?"""
    entropy = -(probs * np.log(probs)).sum(1)
    grade_std = grade_passes.std(0)
    err = (pred != y).astype(int)
    res = {"auroc_entropy_detects_error": float(roc_auc_score(err, entropy)) if 0 < err.sum() < len(err) else None,
           "auroc_gradestd_detects_error": float(roc_auc_score(err, grade_std)) if 0 < err.sum() < len(err) else None,
           "mean_entropy_correct": float(entropy[err == 0].mean()),
           "mean_entropy_wrong": float(entropy[err == 1].mean()) if err.sum() else None,
           "referral": []}
    order = np.argsort(-entropy)                      # most uncertain first
    for rate in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        keep = np.sort(order[int(round(rate * len(y))):])
        yk, pk = y[keep], pred[keep]
        res["referral"].append({"referral_rate": rate, "n_kept": int(len(keep)),
                                "accuracy": float(accuracy_score(yk, pk)),
                                "qwk": float(cohen_kappa_score(yk, pk, weights="quadratic"))})
    save_json(res, os.path.join(run_dir, "uncertainty.json"))

    r = pd.DataFrame(res["referral"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(r.referral_rate * 100, r.accuracy * 100, "o-"); ax[0].set_ylabel("Accuracy on retained (%)")
    ax[1].plot(r.referral_rate * 100, r.qwk, "o-", color="#c05621"); ax[1].set_ylabel("QWK on retained")
    for a in ax: a.set_xlabel("Most-uncertain cases referred to specialist (%)"); a.grid(alpha=.3)
    fig.suptitle(f"{run}: MC-dropout uncertainty-based referral"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "referral_curve.png"), dpi=150); plt.close()
    return res, entropy, grade_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--img-dir", default=None, help="Defaults to the folder used in training")
    ap.add_argument("--out", default=C.OUTPUT_DIR)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--mc-samples", type=int, default=0, help="MC-dropout passes (e.g. 30); 0 = off")
    args = ap.parse_args()

    device = get_device()
    run_dir = os.path.join(args.out, args.run)
    model, ckpt = load_model(os.path.join(run_dir, "best_model.pt"), device)
    head = ckpt["cfg"]["head"]
    img_dir = args.img_dir or ckpt["img_dir"]

    split = pd.read_csv(os.path.join(args.out, "split.csv"))
    te_df = split[split.split == "test"].reset_index(drop=True)
    ds = DRDataset(te_df, img_dir, get_transforms(ckpt["img_size"], train=False))
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=C.NUM_WORKERS)

    # ---- deterministic prediction (dropout off) ----
    outs, y = collect(model, dl, device, 1, tta=not args.no_tta)
    probs, pred, grade = aggregate(list(outs[0]), head)
    probs, pred = probs.numpy(), pred.numpy()
    m = compute_metrics(y, pred, C.NUM_CLASSES)
    m.update(run=args.run, n_params=ckpt["n_params"], best_epoch=ckpt["epoch"], head=head, tta=not args.no_tta)
    preds_df = te_df[["id_code", "diagnosis"]].assign(pred=pred, **{f"p{k}": probs[:, k] for k in range(C.NUM_CLASSES)})

    # ---- MC dropout ----
    if args.mc_samples > 0:
        enable_mc_dropout(model)
        outs_mc, _ = collect(model, dl, device, args.mc_samples, tta=False)
        model.eval()
        pass_outs = [o[0] for o in outs_mc]                       # list of (N, out_dim)
        p_mc, pred_mc, _ = aggregate(pass_outs, head)
        grade_passes = np.stack([aggregate([o], head)[2].numpy() for o in pass_outs])
        p_mc, pred_mc = p_mc.numpy(), pred_mc.numpy()
        m_mc = compute_metrics(y, pred_mc, C.NUM_CLASSES)
        res, ent, gstd = uncertainty_analysis(y, pred_mc, p_mc, grade_passes, run_dir, args.run)
        m["mc_dropout"] = {"samples": args.mc_samples, "accuracy": m_mc["accuracy"], "qwk": m_mc["qwk"],
                           "f1_macro": m_mc["f1_macro"], **{k: v for k, v in res.items() if k != "referral"}}
        preds_df = preds_df.assign(pred_mc=pred_mc, entropy=ent, grade_std=gstd)
        print(f"MC-dropout ({args.mc_samples} passes): acc {m_mc['accuracy']:.4f} | QWK {m_mc['qwk']:.4f}")
        print(f"Uncertainty detects errors: AUROC {res['auroc_entropy_detects_error']}")
        for r in res["referral"]:
            print(f"  refer {r['referral_rate']*100:4.0f}% -> acc {r['accuracy']:.4f} | QWK {r['qwk']:.4f}")

    save_json(m, os.path.join(run_dir, "metrics.json"))
    preds_df.to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)
    report = classification_report(y, pred, labels=list(range(C.NUM_CLASSES)),
                                   target_names=C.CLASS_NAMES, digits=4, zero_division=0)
    with open(os.path.join(run_dir, "classification_report.txt"), "w") as f:
        f.write(report + f"\nQuadratic Weighted Kappa: {m['qwk']:.4f}\n")
    plot_confusion(m["confusion_matrix"], os.path.join(run_dir, "confusion_matrix.png"),
                   f"{args.run} — QWK {m['qwk']:.3f}")
    print(report)
    print(f"[{args.run}] Acc {m['accuracy']:.4f} | P {m['precision_macro']:.4f} | R {m['recall_macro']:.4f} "
          f"| F1 {m['f1_macro']:.4f} | QWK {m['qwk']:.4f} | minority recall {m['minority_recall']:.4f}")


if __name__ == "__main__":
    main()
