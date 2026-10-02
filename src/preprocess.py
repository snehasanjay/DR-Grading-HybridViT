"""
Fundus image preprocessing. Two methods:
  graham : base paper (LSD-HybridViT) style - crop, resize, Gaussian local-average
           subtraction ("controlled smoothing", the grey images in its Fig. 1)
  clahe  : ours - crop, resize, CLAHE on the LAB L-channel (keeps colour, boosts
           local lesion contrast)
Both finish with a circular mask.

Run once per method before training:
    python -m src.preprocess --method graham --dst /kaggle/working/prep_graham
    python -m src.preprocess --method clahe  --dst /kaggle/working/prep_clahe
"""
import os, argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from . import config as C


def crop_black_borders(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def apply_clahe(img, clip_limit=2.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def graham_enhance(img, sigma_frac=1 / 30):
    """Ben Graham's local average colour subtraction: 4*img - 4*blur(img) + 128."""
    sigma = img.shape[0] * sigma_frac
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def circular_mask(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 1, -1)
    return img * mask[..., None]


def preprocess_image(path, size=C.IMG_SIZE, method="clahe"):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_black_borders(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = apply_clahe(img) if method == "clahe" else graham_enhance(img)
    img = circular_mask(img)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=C.TRAIN_CSV)
    ap.add_argument("--src", default=C.RAW_IMG_DIR)
    ap.add_argument("--dst", default=C.PREPROC_DIR)
    ap.add_argument("--size", type=int, default=C.IMG_SIZE)
    ap.add_argument("--method", choices=["clahe", "graham"], default="clahe")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    df = pd.read_csv(args.csv)
    for id_code in tqdm(df["id_code"], desc=f"Preprocessing ({args.method})"):
        out = os.path.join(args.dst, f"{id_code}.png")
        if os.path.exists(out):
            continue
        img = preprocess_image(os.path.join(args.src, f"{id_code}.png"), args.size, args.method)
        cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Done. Saved {len(df)} images to {args.dst}")


if __name__ == "__main__":
    main()
