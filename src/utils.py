import os, random, json
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             cohen_kappa_score, confusion_matrix)


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true, y_pred, num_classes=5):
    """All metrics named in the abstract: accuracy, precision, recall, F1, QWK."""
    labels = list(range(num_classes))
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pc_p, pc_r, pc_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return {
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "qwk": float(qwk),
        "per_class": {
            "precision": pc_p.tolist(), "recall": pc_r.tolist(),
            "f1": pc_f1.tolist(), "support": support.tolist(),
        },
        # recall on Severe (3) + Proliferative (4): the minority classes the abstract highlights
        "minority_recall": float(np.mean(pc_r[3:5])),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
