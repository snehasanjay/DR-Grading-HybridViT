#!/usr/bin/env bash
# Full ablation study: base LSD-HybridViT reproduction -> + each enhancement.
# Usage (Kaggle):  DR_DATA_DIR=... bash run_experiments.sh
# Optional env:    PREP=/kaggle/working  EPOCHS=30  SIZE=384
set -e
PREP=${PREP:-/kaggle/working}
EPOCHS=${EPOCHS:-30}
SIZE=${SIZE:-384}
G=$PREP/prep_graham
L=$PREP/prep_clahe

echo "===== 1. Preprocessing ====="
python -m src.preprocess --method graham --size $SIZE --dst $G
python -m src.preprocess --method clahe  --size $SIZE --dst $L

echo "===== 2. Training ====="
python -m src.train --run A_base       --img-dir $G --no-pretrained --epochs $EPOCHS --img-size $SIZE
python -m src.train --run B_pretrained --img-dir $G                 --epochs $EPOCHS --img-size $SIZE
python -m src.train --run C_clahe      --img-dir $L                 --epochs $EPOCHS --img-size $SIZE
python -m src.train --run D_ordinal    --img-dir $L --loss ordinal  --epochs $EPOCHS --img-size $SIZE

echo "===== 3. Evaluation on test set ====="
for R in A_base B_pretrained C_clahe; do python -m src.evaluate --run $R; done
python -m src.evaluate --run D_ordinal --mc-samples 30      # full model + MC-dropout uncertainty

echo "===== 4. Figures ====="
python -m src.compare --runs A_base B_pretrained C_clahe D_ordinal
python -m src.gradcam --run D_ordinal --compare A_base --n 5
python -m src.gradcam --run D_ordinal --n 10
python -m src.figure_preprocess --graham-dir $G --clahe-dir $L

mkdir -p paper_figures
cp outputs/preprocessing_examples.png outputs/training_curves.png outputs/ablation_metrics.png paper_figures/
cp outputs/A_base/confusion_matrix.png    paper_figures/confusion_base.png
cp outputs/D_ordinal/confusion_matrix.png paper_figures/confusion_ours.png
cp outputs/D_ordinal/referral_curve.png   paper_figures/referral_curve.png
cp outputs/gradcam_D_ordinal_vs_A_base.png paper_figures/gradcam_compare.png
cd outputs && zip -rq ../results.zip . -x "*.pt" && cd .. && zip -rq results.zip paper_figures
echo "===== DONE: download results.zip ====="
