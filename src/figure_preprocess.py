"""
Paper figure: one image per DR grade - raw, base-paper (Graham) preprocessing, ours (CLAHE).
    python -m src.figure_preprocess --graham-dir /kaggle/working/prep_graham --clahe-dir /kaggle/working/prep_clahe
"""
import os, argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from . import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=C.TRAIN_CSV)
    ap.add_argument("--raw-dir", default=C.RAW_IMG_DIR)
    ap.add_argument("--graham-dir", required=True)
    ap.add_argument("--clahe-dir", required=True)
    ap.add_argument("--out", default=C.OUTPUT_DIR)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    picks = df.groupby("diagnosis").sample(1, random_state=1).sort_values("diagnosis")
    rows = [("Raw", args.raw_dir), ("Base: Graham", args.graham_dir), ("Ours: CLAHE", args.clahe_dir)]
    fig, axes = plt.subplots(3, 5, figsize=(15, 9.4))
    for j, (_, r) in enumerate(picks.iterrows()):
        for i, (label, d) in enumerate(rows):
            axes[i, j].imshow(Image.open(os.path.join(d, f"{r.id_code}.png")).convert("RGB"))
            axes[i, j].axis("off")
        axes[0, j].set_title(C.CLASS_NAMES[r.diagnosis], fontsize=13)
    for i, (label, _) in enumerate(rows):
        axes[i, 0].text(-0.08, 0.5, label, transform=axes[i, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=13)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "preprocessing_examples.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print("Saved", out)


if __name__ == "__main__":
    main()
