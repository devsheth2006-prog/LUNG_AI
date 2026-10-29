# models/

Empty on purpose. Each training script writes `models/<task>/model.pt|model.joblib` plus a
`model_card.json` containing the architecture, datasets, patient counts and the **held-out test
metrics computed by the evaluation code**. The app and report read only these cards; a task
without a card is shown as **"Model not trained"**.

| task folder | written by |
|---|---|
| `lung_segmentation/` | `training/train_segmentation.py --target lung` |
| `tumor_segmentation/` | `training/train_segmentation.py --target tumor` |
| `nodule_detection/` | `training/train_classifier.py --task nodule_detection` |
| `malignancy/` | `training/train_classifier.py --task malignancy` |
| `subtype/`, `t_category/` | `training/train_stage_model.py` |
| `n_involvement/`, `m_distant/`, `distant_recurrence/` | `training/train_metastasis_model.py` |
| `classical_study/` | `training/run_pca_lda.py` |
| `multimodal_study/` | `training/train_multimodal.py` |
