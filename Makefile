PY ?= python
.PHONY: install test smoke app api data-lidc train-all summary docker

install:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt && pip install -e .

test:
	$(PY) -m pytest tests

smoke:            ## every training script on a synthetic fixture (software check only)
	bash tests/integration/run_training_smoke.sh

app:
	streamlit run app/Home.py

api:
	uvicorn api.main:app --reload --port 8000

train-all:        ## full pipeline on real data (after downloading datasets, see docs/DATASETS.md)
	$(PY) training/prepare_data.py --dataset luna16_lung
	$(PY) training/prepare_data.py --dataset luna16_candidates
	$(PY) training/prepare_data.py --dataset lidc
	$(PY) training/prepare_data.py --dataset nsclc_radiomics
	$(PY) training/prepare_data.py --dataset nsclc_radiogenomics
	$(PY) training/train_segmentation.py --target lung
	$(PY) training/train_segmentation.py --target tumor
	$(PY) training/train_classifier.py --task nodule_detection
	$(PY) training/train_classifier.py --task malignancy
	$(PY) training/extract_features.py --dataset lidc
	$(PY) training/extract_features.py --dataset nsclc_radiomics
	$(PY) training/extract_features.py --dataset nsclc_radiogenomics
	$(PY) training/run_pca_lda.py
	$(PY) training/train_stage_model.py
	$(PY) training/train_metastasis_model.py
	$(PY) training/train_multimodal.py
	$(PY) evaluation/evaluate.py

summary:
	$(PY) evaluation/evaluate.py

docker:
	docker build -t ct-cancer-ai .
