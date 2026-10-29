#!/usr/bin/env bash
# End-to-end software smoke test of every training script on a SYNTHETIC fixture (meaningless metrics).
# Usage: bash tests/integration/run_training_smoke.sh [workdir]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
W="${1:-$(mktemp -d)}"
export PYTHONPATH="$ROOT/src:$ROOT/tests/integration"
python -c "from synthetic_fixture import build; from pathlib import Path; build(Path('$W'))"
S=(--set "paths.processed_root=$W/processed" --set "paths.splits_dir=$W/splits" --set "paths.features_dir=$W/features"
   --set "paths.models_dir=$W/models" --set "paths.figures_dir=$W/figures" --set "paths.mlruns=$W/mlruns"
   --set training.epochs=2 --set training.batch_size=8 --set training.num_workers=0 --set training.freeze_backbone_epochs=1
   --set classifier.pretrained=false --set evaluation.bootstrap_iterations=50 --set risk.min_positive_labels=5
   --set "classifier.architectures=[resnet18,densenet121]" --set preprocessing.slice_size=96)
cd "$ROOT"
python training/train_segmentation.py --target lung "${S[@]}"
python training/train_segmentation.py --target tumor "${S[@]}"
python training/train_classifier.py --task nodule_detection "${S[@]}"
python training/train_classifier.py --task malignancy "${S[@]}"
python training/extract_features.py --dataset lidc "${S[@]}"
python training/extract_features.py --dataset nsclc_radiomics "${S[@]}"
python training/run_pca_lda.py "${S[@]}"
python training/train_stage_model.py "${S[@]}"
python training/train_metastasis_model.py "${S[@]}"
python training/train_multimodal.py --epochs 30 "${S[@]}"
python evaluation/evaluate.py "${S[@]}"
echo "SMOKE OK: $W"
