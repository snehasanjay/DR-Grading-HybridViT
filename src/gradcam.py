"""
Grad-CAM explanations on the final feature map of the network.

Grid of test images (spread across grades):  python -m src.gradcam --run D_ordinal --n 10
Side-by-side comparison of two runs:          python -m src.gradcam --run D_ordinal --compare A_base
Single raw image:                             python -m src.gradcam --run D_ordinal --image fundus.png
"""
import os, argparse
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from . import config as C
from .dataset import get_transforms
from .evaluate import load_model
from .models import outputs_to_probs, outputs_to_pred, cam_score
from .preprocess import preprocess_image
from .utils import get_device


class GradCAM:
    def __init__(self, model):
        self.model, self.head = model, model.head_type
        self.acts = self.grads = None
        model.cam_layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x):
        self.model.zero_grad()
        out = self.model(x)
        probs = outputs_to_probs(out.detach(), self.head)[0].cpu().numpy()
        pred = int(outputs_to_pred(out.detach(), self.head)[0])
        cam_score(out, self.head, pred).backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        raw = (w * self.acts).sum(1)[0]
        cam = torch.relu(raw)
        if cam.max() <= 0:          # no positive evidence at all: show relative evidence instead
            cam = raw - raw.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), pred, probs


def overlay(img, cam, alpha=0.4):
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * img + alpha * heat), cam


def explain(cam_fn, img, size, device):
    x = get_transforms(size, train=False)(Image.fromarray(img)).unsqueeze(0).to(device)
    cam, pred, probs = cam_fn(x)
    img_s = cv2.resize(img, (size, size))
    over, cam_up = overlay(img_s, cam)
    return img_s, cam_up, over, pred, probs


def load(run, out, device):
    model, ckpt = load_model(os.path.join(out, run, "best_model.pt"), device)
    return model, ckpt, GradCAM(model)


def pick_test_images(out, n, seed=0):
    split = pd.read_csv(os.path.join(out, "split.csv"))
    te = split[split.split == "test"]
    per = max(1, n // C.NUM_CLASSES)
    return pd.concat([g.sample(min(len(g), per), random_state=seed) for _, g in te.groupby("diagnosis")]).head(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare", default=None, help="Second run to show side by side")
    ap.add_argument("--out", default=C.OUTPUT_DIR)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--image", default=None)
    args = ap.parse_args()
    device = get_device()
    model, ckpt, cam_fn = load(args.run, args.out, device)
    size = ckpt["img_size"]

    if args.image:
        method = "clahe" if "clahe" in ckpt["img_dir"] else "graham"
        img = preprocess_image(args.image, size, method)
        img_s, cam, over, pred, probs = explain(cam_fn, img, size, device)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_s); ax[0].set_title("Preprocessed fundus")
        ax[1].imshow(over); ax[1].set_title(f"Pred: {C.CLASS_NAMES[pred]} ({probs[pred]:.1%})")
        for a in ax: a.axis("off")
        out = os.path.join(args.out, args.run, f"gradcam_{os.path.splitext(os.path.basename(args.image))[0]}.png")
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print({C.CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)}); print("Saved", out)
        return

    picks = pick_test_images(args.out, args.n)
    runs = [(args.run, cam_fn, ckpt)]
    if args.compare:
        m2, ck2, cam2 = load(args.compare, args.out, device)
        runs.append((args.compare, cam2, ck2))
    cols = 1 + 2 * len(runs)
    fig, axes = plt.subplots(len(picks), cols, figsize=(3.2 * cols, 3.3 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, (_, r) in zip(axes, picks.iterrows()):
        for i, (name, fn, ck) in enumerate(runs):
            img = np.array(Image.open(os.path.join(ck["img_dir"], f"{r.id_code}.png")).convert("RGB"))
            img_s, cam, over, pred, probs = explain(fn, img, ck["img_size"], device)
            if i == 0:
                row[0].imshow(img_s); row[0].set_title(f"True: {C.CLASS_NAMES[r.diagnosis]}", fontsize=10)
            ok = "✓" if pred == r.diagnosis else "✗"
            row[1 + 2 * i].imshow(cam, cmap="jet"); row[1 + 2 * i].set_title(f"{name}: Grad-CAM", fontsize=9)
            row[2 + 2 * i].imshow(over)
            row[2 + 2 * i].set_title(f"{name}: {C.CLASS_NAMES[pred]} ({probs[pred]:.0%}) {ok}", fontsize=9)
        for a in row: a.axis("off")
    name = f"gradcam_{args.run}_vs_{args.compare}.png" if args.compare else "gradcam_grid.png"
    out = os.path.join(args.out, name if args.compare else os.path.join(args.run, name))
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print("Saved", out)


if __name__ == "__main__":
    main()
