"""
Demo: grade one fundus photograph with a trained model.

    python -m src.predict --ckpt weights/D_ordinal.pt --image my_fundus.png

Prints the predicted DR grade, class probabilities, MC-dropout uncertainty and a
referral recommendation, and saves a figure with the preprocessed image and Grad-CAM.
Runs on CPU (about 10-30 s per image) or GPU.
"""
import os, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from . import config as C
from .dataset import get_transforms
from .evaluate import load_model, aggregate
from .gradcam import GradCAM, overlay
from .models import enable_mc_dropout
from .preprocess import preprocess_image
from .utils import get_device

# Mean predictive entropy on the APTOS test set was 0.54 for correct and 1.33 for wrong
# predictions; 1.0 is a simple heuristic cut-off between the two, not a tuned threshold.
DEFAULT_ENTROPY_THRESHOLD = 1.0


@torch.no_grad()
def mc_predict(model, x, head, passes):
    enable_mc_dropout(model)
    outs = [model(x).float().cpu() for _ in range(passes)]
    model.eval()
    probs, pred, _ = aggregate(outs, head)
    p = probs[0].numpy()
    entropy = float(-(p * np.log(p)).sum())
    return int(pred[0]), p, entropy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best_model.pt (e.g. weights/D_ordinal.pt)")
    ap.add_argument("--image", required=True, help="Raw fundus photograph (png/jpg)")
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--entropy-threshold", type=float, default=DEFAULT_ENTROPY_THRESHOLD)
    ap.add_argument("--out", default="demo_output")
    args = ap.parse_args()

    device = get_device()
    model, ckpt = load_model(args.ckpt, device)
    head, size = ckpt["cfg"]["head"], ckpt["img_size"]
    method = "graham" if "graham" in str(ckpt.get("img_dir", "")) else "clahe"

    img = preprocess_image(args.image, size, method)
    x = get_transforms(size, train=False)(Image.fromarray(img)).unsqueeze(0).to(device)

    pred, probs, entropy = mc_predict(model, x, head, args.mc_samples)
    cam, _, _ = GradCAM(model)(x.clone().requires_grad_(True))
    over, _ = overlay(img, cam)

    uncertain = entropy > args.entropy_threshold
    referable = pred >= 2
    if referable:
        advice = "REFER: referable DR (grade >= Moderate) predicted"
    elif uncertain:
        advice = "REFER: prediction uncertain, specialist review recommended"
    else:
        advice = "No referable DR predicted; routine re-screening"

    print("\n================ RESULT ================")
    print(f"Image            : {args.image}")
    print(f"Predicted grade  : {pred} - {C.CLASS_NAMES[pred]}")
    print("Probabilities    : " + ", ".join(f"{n} {p:.1%}" for n, p in zip(C.CLASS_NAMES, probs)))
    print(f"Uncertainty      : entropy {entropy:.2f} ({'HIGH' if uncertain else 'low'}; "
          f"threshold {args.entropy_threshold})")
    print(f"Recommendation   : {advice}")
    print("Note: research prototype, not a medical device.")
    print("========================================\n")

    os.makedirs(args.out, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    ax[0].imshow(img); ax[0].set_title("Preprocessed fundus (CLAHE)"); ax[0].axis("off")
    ax[1].imshow(over); ax[1].set_title(f"Grad-CAM - predicted: {C.CLASS_NAMES[pred]}"); ax[1].axis("off")
    colors = ["#2f855a" if i == pred else "#a0aec0" for i in range(len(probs))]
    ax[2].barh(C.CLASS_NAMES, probs * 100, color=colors); ax[2].invert_yaxis()
    ax[2].set_xlim(0, 100); ax[2].set_xlabel("Probability (%)")
    ax[2].set_title(f"Uncertainty: {entropy:.2f} ({'HIGH' if uncertain else 'low'})")
    fig.suptitle(advice, fontsize=13, color="#c53030" if (referable or uncertain) else "#2f855a")
    plt.tight_layout()
    out = os.path.join(args.out, f"result_{os.path.splitext(os.path.basename(args.image))[0]}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print("Saved figure:", out)


if __name__ == "__main__":
    main()
