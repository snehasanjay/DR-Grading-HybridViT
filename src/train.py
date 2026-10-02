"""
Train one experiment configuration.

Ablation used in the paper (see run_experiments.sh):
  A  base      : LSD-HybridViT reproduction, Graham preprocessing, from scratch, CE loss
  B  +pre      : A + ImageNet-pretrained MobileViT
  C  +clahe    : B + CLAHE preprocessing
  D  +ordinal  : C + ordinal (CORAL) loss          -> full enhanced model
  (MC-dropout uncertainty is applied to D at test time by evaluate.py, no retraining)

Example:
  python -m src.train --run D_ordinal --img-dir /kaggle/working/prep_clahe --loss ordinal
"""
import os, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import config as C
from .dataset import DRDataset, get_transforms, make_balanced_sampler
from .models import build_model, ordinal_targets, outputs_to_pred
from .utils import seed_everything, get_device, compute_metrics, save_json


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Experiment name, e.g. A_base")
    ap.add_argument("--img-dir", required=True, help="Folder of preprocessed images")
    ap.add_argument("--csv", default=C.TRAIN_CSV)
    ap.add_argument("--loss", choices=["ce", "ordinal"], default="ce")
    ap.add_argument("--no-pretrained", action="store_true", help="Train backbone from scratch")
    ap.add_argument("--no-lma", action="store_true")
    ap.add_argument("--no-jdpm", action="store_true")
    ap.add_argument("--lma-kernel", type=int, default=None, help="Fixed K instead of adaptive")
    ap.add_argument("--img-size", type=int, default=C.IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY)
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    ap.add_argument("--no-balance", action="store_true", help="Disable minority oversampling")
    ap.add_argument("--num-workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--out", default=C.OUTPUT_DIR)
    ap.add_argument("--seed", type=int, default=C.SEED)
    return ap.parse_args()


def make_split(csv, out_dir, seed):
    """Stratified 70/15/15 train/val/test split, saved so every run uses the same images.
    Validation picks the best epoch; the test set is only touched by evaluate.py."""
    path = os.path.join(out_dir, "split.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    df = pd.read_csv(csv)
    tr, rest = train_test_split(df, test_size=0.30, stratify=df["diagnosis"], random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=rest["diagnosis"], random_state=seed)
    split = pd.concat([tr.assign(split="train"), va.assign(split="val"),
                       te.assign(split="test")]).reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    split.to_csv(path, index=False)
    return split


class Criterion(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.ce = nn.CrossEntropyLoss(label_smoothing=C.LABEL_SMOOTHING)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, out, y):
        if self.head == "ce":
            return self.ce(out, y)
        return self.bce(out, ordinal_targets(y, C.NUM_CLASSES))


def run_epoch(model, loader, criterion, device, head, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total, preds, targets = 0.0, [], []
    use_amp = scaler is not None
    for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=use_amp):
            out = model(x)
            loss = criterion(out.float(), y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        total += loss.item() * x.size(0)
        preds.append(outputs_to_pred(out.float(), head).cpu()); targets.append(y.cpu())
    return total / len(loader.dataset), torch.cat(targets).numpy(), torch.cat(preds).numpy()


def main():
    args = get_args()
    seed_everything(args.seed)
    device = get_device()
    run_dir = os.path.join(args.out, args.run)
    os.makedirs(run_dir, exist_ok=True)
    cfg = dict(pretrained=not args.no_pretrained, use_lma=not args.no_lma,
               use_jdpm=not args.no_jdpm, head=args.loss, dropout=C.DROPOUT,
               lma_kernel=args.lma_kernel)
    print(f"Device: {device} | Run: {args.run} | {cfg} | images: {args.img_dir}")

    split = make_split(args.csv, args.out, args.seed)
    tr_df, va_df = split[split.split == "train"], split[split.split == "val"]
    tr_ds = DRDataset(tr_df, args.img_dir, get_transforms(args.img_size, train=True))
    va_ds = DRDataset(va_df, args.img_dir, get_transforms(args.img_size, train=False))
    sampler = None if args.no_balance else make_balanced_sampler(tr_df.diagnosis.values, C.NUM_CLASSES)
    pin = device.type == "cuda"
    tr_dl = DataLoader(tr_ds, args.batch_size, sampler=sampler, shuffle=sampler is None,
                       num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    va_dl = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    model = build_model(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.2f}M")

    criterion = Criterion(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=1, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_qwk, bad, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_y, tr_p = run_epoch(model, tr_dl, criterion, device, args.loss, optimizer, scaler)
        va_loss, va_y, va_p = run_epoch(model, va_dl, criterion, device, args.loss)
        scheduler.step()
        tr_m, va_m = compute_metrics(tr_y, tr_p), compute_metrics(va_y, va_p)
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss,
                            train_acc=tr_m["accuracy"], val_acc=va_m["accuracy"],
                            train_qwk=tr_m["qwk"], val_qwk=va_m["qwk"], val_f1=va_m["f1_macro"],
                            lr=optimizer.param_groups[0]["lr"]))
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        print(f"Epoch {epoch:02d} | loss {tr_loss:.4f}/{va_loss:.4f} | acc {va_m['accuracy']:.4f} "
              f"| QWK {va_m['qwk']:.4f} | F1 {va_m['f1_macro']:.4f} | {time.time()-t0:.0f}s", flush=True)
        if va_m["qwk"] > best_qwk:
            best_qwk, bad = va_m["qwk"], 0
            torch.save({"model_state": model.state_dict(), "cfg": cfg, "img_size": args.img_size,
                        "img_dir": args.img_dir, "epoch": epoch, "val_qwk": best_qwk,
                        "n_params": n_params}, os.path.join(run_dir, "best_model.pt"))
            print(f"  -> saved new best (QWK {best_qwk:.4f})", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch}"); break

    save_json({"best_val_qwk": best_qwk, "n_params": n_params, "cfg": cfg, "args": vars(args)},
              os.path.join(run_dir, "train_summary.json"))
    print(f"Finished {args.run}. Best validation QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    main()
