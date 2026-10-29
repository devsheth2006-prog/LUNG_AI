"""End-to-end inference: upload -> validation -> preprocessing -> lung segmentation ->
lesion candidates -> lesion segmentation -> features -> malignancy -> subtype / TNM / spread
-> explainability. Every stage reports *how* its output was produced (trained model, classical
baseline, or unavailable) so the UI can never present an untrained output as a prediction.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import DISCLAIMER, __version__
from ..config import Config, load_config, resolve_device
from ..detection.candidates import full_mask, generate_candidates, nms_by_distance
from ..features.lesion import describe_location, measure_lesion
from ..features.radiomics import extract_radiomics, shape_features
from ..io.loaders import load_any
from ..io.validation import ScanValidationError
from ..io.volume import CTVolume
from ..logging_utils import get_logger
from ..preprocessing.ct import apply_window, extract_patch, preprocess_volume, resize_slice
from ..risk.tasks import TASKS, risk_band
from ..segmentation.lung import classical_lung_mask, lung_volume_ml, split_left_right
from .registry import TaskStatus, all_statuses, models_dir

log = get_logger(__name__)
PATIENT_TASKS = ["subtype", "t_category", "n_involvement", "m_distant", "distant_recurrence"]


@dataclass
class AnalysisArrays:
    """Large arrays for visualisation (kept out of the JSON result)."""
    norm: np.ndarray                     # lung-window [0,1] volume (resampled)
    hu: np.ndarray
    lung_lr: np.ndarray                  # 0 bg / 1 right / 2 left
    lesion_labels: np.ndarray            # 0 bg / k lesion id
    spacing: tuple
    cams: dict[int, dict] = field(default_factory=dict)  # lesion id -> {z, cam2d, cam_pp2d, box}


@dataclass
class AnalysisResult:
    ok: bool
    scan: dict = field(default_factory=dict)
    preprocessing: list[str] = field(default_factory=list)
    lung: dict = field(default_factory=dict)
    lesions: list[dict] = field(default_factory=list)
    index_lesion: int | None = None
    patient_predictions: dict[str, dict] = field(default_factory=dict)
    statuses: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    runtime_s: float = 0.0
    software_version: str = __version__
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


def _jsonable(o: Any):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


class ModelBank:
    """Lazily loads trained models; missing/corrupted models degrade gracefully."""

    def __init__(self, cfg: Config, root: Path | None = None):
        self.cfg, self.root = cfg, Path(root) if root else cfg.path("models_dir")
        self.statuses: dict[str, TaskStatus] = all_statuses(self.root)
        self._cache: dict[str, Any] = {}
        self.device = None

    def _dev(self):
        if self.device is None:
            self.device = resolve_device(self.cfg.get("device", "auto"))
        return self.device

    def torch_model(self, task: str):
        st = self.statuses[task]
        if not st.usable:
            return None
        if task in self._cache:
            return self._cache[task]
        try:
            import torch

            from ..models.classifiers import build_classifier
            from ..models.unet import build_segmentation_model

            card = st.card
            ckpt = torch.load(self.root / task / "model.pt", map_location="cpu", weights_only=False)
            if task in ("lung_segmentation", "tumor_segmentation"):
                model = build_segmentation_model(card["architecture"], 1, 2, card.get("base_channels", 16))
            else:
                model = build_classifier(card["architecture"], len(card.get("classes", [0, 1])), pretrained=False)
            model.load_state_dict(ckpt["state_dict"])
            model.eval().to(self._dev())
            self._cache[task] = (model, card)
        except Exception as exc:  # corrupted weights, version mismatch, torch missing
            log.error("Failed to load %s: %s", task, exc)
            self.statuses[task] = TaskStatus(task, "not_trained", f"Model failed to load ({type(exc).__name__}).")
            self._cache[task] = None
        return self._cache[task]

    def tabular_model(self, task: str):
        st = self.statuses[task]
        if not st.usable:
            return None
        if task not in self._cache:
            try:
                import joblib

                self._cache[task] = joblib.load(self.root / task / "model.joblib")
            except Exception as exc:
                log.error("Failed to load %s: %s", task, exc)
                self.statuses[task] = TaskStatus(task, "not_trained", f"Model failed to load ({type(exc).__name__}).")
                self._cache[task] = None
        return self._cache[task]


class AnalysisPipeline:
    def __init__(self, cfg: Config | None = None, models_root: Path | None = None):
        self.cfg = cfg or load_config()
        self.bank = ModelBank(self.cfg, models_root)

    # ----------------------------------------------------------------- public API
    def run(self, path_or_volume: str | Path | CTVolume, clinical: dict | None = None,
            keep_arrays: bool = True) -> tuple[AnalysisResult, AnalysisArrays | None]:
        t0 = time.time()
        res = AnalysisResult(ok=False, statuses={k: {"state": v.state, "badge": v.badge, "message": v.message}
                                                 for k, v in self.bank.statuses.items()})
        try:
            vol = path_or_volume if isinstance(path_or_volume, CTVolume) else \
                load_any(path_or_volume, self.cfg.get_path("app.max_upload_mb", 500))
            res.scan = vol.summary()
            res.warnings += vol.warnings
            arrays = self._analyse(vol, res, clinical or {})
            res.ok = True
        except ScanValidationError as exc:
            res.error = str(exc)
            arrays = None
        except MemoryError:
            res.error = "The scan is too large to process with the available memory. Try the lite profile."
            arrays = None
        except Exception as exc:  # last-resort guard: never show a traceback to the user
            log.error("Unexpected failure: %s\n%s", exc, traceback.format_exc())
            res.error = f"Processing failed unexpectedly ({type(exc).__name__}). The scan may be malformed."
            arrays = None
        res.runtime_s = round(time.time() - t0, 2)
        return res, (arrays if keep_arrays else None)

    # ----------------------------------------------------------------- stages
    def _analyse(self, vol: CTVolume, res: AnalysisResult, clinical: dict) -> AnalysisArrays:
        cfg = self.cfg
        pre = preprocess_volume(vol, cfg["preprocessing"])
        res.preprocessing = pre.steps
        sp = pre.spacing

        # ---- lung segmentation
        lung, method = self._lung_mask(pre)
        lr = split_left_right(lung)
        res.lung = {"method": method, "volume_ml": round(lung_volume_ml(lung, sp), 1) if vol.quantitative else None,
                    "components": int(len(np.unique(lr)) - 1)}
        if not lung.any():
            res.warnings.append("No lung region was found; lesion analysis was skipped. Is this a chest CT?")
            return AnalysisArrays(pre.norm, pre.hu, lr, np.zeros(lr.shape, np.int16), sp)

        # ---- candidates + detection scoring
        dcfg = cfg["detection"]
        cands = generate_candidates(pre.hu, lung, sp, min_d_mm=dcfg["candidate_min_diameter_mm"],
                                    max_d_mm=dcfg["candidate_max_diameter_mm"],
                                    max_candidates=dcfg["max_candidates"], is_hu=pre.is_hu)
        cands = nms_by_distance(cands, sp, dcfg["nms_distance_mm"])
        det = self.bank.torch_model("nodule_detection") if pre.is_hu else None
        det_probs = self._score_candidates(det, pre.hu, cands) if det else None
        if det_probs is not None:
            thr = det[1].get("threshold", dcfg["score_threshold"])
            keep = [i for i, p in enumerate(det_probs) if p >= thr]
            cands = [cands[i] for i in keep]
            det_probs = [det_probs[i] for i in keep]

        labels = np.zeros(lr.shape, np.int16)
        arrays = AnalysisArrays(pre.norm, pre.hu, lr, labels, sp)
        mal_model = self.bank.torch_model("malignancy") if pre.is_hu else None
        for i, c in enumerate(cands):
            lesion = self._lesion(c, pre, lung, lr, labels, mal_model, arrays, vol.quantitative)
            lesion["detection"] = {
                "method": "CNN false-positive reduction" if det_probs is not None else "classical candidate generator",
                "probability": round(float(det_probs[i]), 4) if det_probs is not None else None,
                "blobness_index": c.blobness,
                "note": None if det_probs is not None else
                ("Unscored candidate: detection model not trained; blobness is a geometric heuristic, not a probability."),
            }
            res.lesions.append(lesion)
        if not res.lesions:
            res.warnings.append("No suspicious lesion candidates were detected within the lungs.")

        # ---- index lesion: highest detection prob, else largest
        if res.lesions:
            key = (lambda l: l["detection"]["probability"] or 0) if det_probs is not None else \
                (lambda l: l["measurements"]["volume_mm3"] or 0)
            res.index_lesion = max(res.lesions, key=key)["id"]
            idx = next(l for l in res.lesions if l["id"] == res.index_lesion)
            res.patient_predictions = self._patient_tasks(idx, clinical, pre.is_hu)
        else:
            res.patient_predictions = {t: self._unavailable(t, "No lesion to characterise.") for t in PATIENT_TASKS}
        for t in ("organ_specific_metastasis", "small_cell"):
            res.patient_predictions[t] = self._unavailable(t, TASKS[t].output_wording, state="unsupported")
        return arrays

    def _lung_mask(self, pre) -> tuple[np.ndarray, str]:
        m = self.bank.torch_model("lung_segmentation") if pre.is_hu else None
        if m is None:
            return classical_lung_mask(pre.hu, pre.spacing, pre.is_hu), \
                "classical HU-threshold baseline (U-Net not trained)"
        import torch

        model, card = m
        size = card.get("input_size", 192)
        out = np.zeros(pre.norm.shape, bool)
        with torch.no_grad():
            for z0 in range(0, pre.norm.shape[0], 16):
                sl = np.stack([resize_slice(s, size) for s in pre.norm[z0:z0 + 16]])
                pred = model(torch.from_numpy(sl[:, None]).float().to(self.bank._dev())).argmax(1).cpu().numpy()
                for k, p in enumerate(pred):
                    out[z0 + k] = _resize_to(p.astype(np.float32), pre.norm.shape[1:]) > 0.5
        return out, f"U-Net ({card.get('architecture')}, v{card.get('version', '?')})"

    def _score_candidates(self, det, hu, cands) -> list[float]:
        import torch

        from ..data.datasets import patch_to_input

        model, card = det
        size = card.get("patch_vox", 48)
        mode = card.get("input_mode", "2.5d")
        xs = [patch_to_input(extract_patch(hu, c.centroid_zyx, size), mode) for c in cands]
        if not xs:
            return []
        with torch.no_grad():
            logits = model(torch.from_numpy(np.stack(xs)).float().to(self.bank._dev()))
            t = card.get("temperature", 1.0)
            return (logits / t).softmax(1)[:, 1].cpu().numpy().tolist()

    def _lesion(self, c, pre, lung, lr, labels, mal_model, arrays, quantitative: bool) -> dict:
        shape = pre.hu.shape
        mask = full_mask(c, shape)
        seg_method = "classical density component (U-Net not trained)"
        seg = self.bank.torch_model("tumor_segmentation") if pre.is_hu else None
        if seg is not None:
            mask = self._segment_lesion(seg, pre, c) | np.zeros(shape, bool)
            seg_method = f"U-Net ({seg[1].get('architecture')})"
            if not mask.any():
                mask = full_mask(c, shape)
                seg_method = "classical fallback (U-Net produced empty mask)"
        labels[mask & (labels == 0)] = c.id
        idx_ = np.argwhere(mask)
        crop = tuple(slice(a, b) for a, b in zip(idx_.min(0), idx_.max(0) + 1))
        shape_f = shape_features(mask[crop], pre.spacing)
        meas = measure_lesion(mask, pre.spacing, shape_f)
        loc = describe_location(meas["centroid_zyx_vox"], lr, lung)
        feats = extract_radiomics(pre.hu, mask, pre.spacing, shape_f) if quantitative else {}
        lesion = {"id": c.id, "bbox_zyx": list(c.bbox_zyx), "segmentation_method": seg_method,
                  "measurements": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in meas.items()}
                  if quantitative else {"note": "Physical measurements unavailable for non-HU input."},
                  "location": loc, "radiomics": {k: round(float(v), 4) for k, v in feats.items()},
                  "malignancy": {"status": self.bank.statuses["malignancy"].badge, "probability": None,
                                 "wording": TASKS["malignancy"].output_wording},
                  "deep_embedding": None}
        if mal_model is not None:
            lesion.update(self._malignancy(mal_model, pre, meas["centroid_zyx_vox"], c.id, arrays))
        return lesion

    def _segment_lesion(self, seg, pre, c) -> np.ndarray:
        import torch

        model, card = seg
        size = card.get("patch_vox", 64)
        cz, cy, cx = (int(round(v)) for v in c.centroid_zyx)
        patch = extract_patch(pre.hu, c.centroid_zyx, size)
        x = apply_window(patch, -600, 1500)
        inp = x[None, None] if card.get("architecture") == "unet3d" else x[:, None]
        with torch.no_grad():
            pred = model(torch.from_numpy(inp).float().to(self.bank._dev())).argmax(1).cpu().numpy().astype(bool)
        pred = pred.reshape(patch.shape)
        out = np.zeros(pre.hu.shape, bool)
        lo = np.array([cz, cy, cx]) - size // 2
        src_lo = np.maximum(lo, 0)
        src_hi = np.minimum(lo + size, pre.hu.shape)
        d_lo = src_lo - lo
        d_hi = d_lo + (src_hi - src_lo)
        out[tuple(slice(a, b) for a, b in zip(src_lo, src_hi))] = pred[tuple(slice(a, b) for a, b in zip(d_lo, d_hi))]
        # keep the connected component containing / nearest the candidate centre
        from skimage import measure

        lab = measure.label(out)
        if lab.max() > 1:
            ids, counts = np.unique(lab[lab > 0], return_counts=True)
            centre_id = lab[cz, cy, cx] if lab[cz, cy, cx] else ids[np.argmax(counts)]
            out = lab == centre_id
        return out

    def _malignancy(self, mal, pre, centroid, lid: int, arrays: AnalysisArrays) -> dict:
        import torch

        from ..data.datasets import patch_to_input
        from ..explain.gradcam import GradCAM

        model, card = mal
        size = card.get("patch_vox", 64)
        mode = card.get("input_mode", "2.5d")
        patch = extract_patch(pre.hu, centroid, size)
        x = torch.from_numpy(patch_to_input(patch, mode, card.get("crop"))[None]).float().to(self.bank._dev())
        with torch.no_grad():
            logits = model(x)
            emb = model.embed(x)[0].cpu().numpy()
        T = card.get("temperature", 1.0)
        p = float((logits / T).softmax(1)[0, 1])
        cams = {}
        for name, pp in (("gradcam", False), ("gradcam_pp", True)):
            gc = GradCAM(model, model.gradcam_layer, plus_plus=pp)
            cam, _, _ = gc(x, class_idx=1)
            gc.remove()
            cams[name] = cam if cam.ndim == 2 else cam[cam.shape[0] // 2]
        arrays.cams[lid] = {"z": int(round(centroid[0])), "centre_yx": (float(centroid[1]), float(centroid[2])),
                            "size": size, "patch": apply_window(patch[size // 2], -600, 1500), **cams}
        return {"malignancy": {"status": self.bank.statuses["malignancy"].badge, "probability": round(p, 4),
                               "benign_probability": round(1 - p, 4), "temperature": T,
                               "wording": TASKS["malignancy"].output_wording,
                               "model": f"{card.get('architecture')} v{card.get('version', '?')}"},
                "deep_embedding": emb.tolist()}

    # ----------------------------------------------------------------- patient-level heads
    def _unavailable(self, task, message, state=None) -> dict:
        st = self.bank.statuses.get(task)
        return {"task": TASKS[task].title, "state": state or (st.state if st else "not_trained"),
                "badge": st.badge if st else "Model not trained", "probabilities": None,
                "message": message, "wording": TASKS[task].output_wording, "caveat": TASKS[task].caveat}

    def _patient_tasks(self, lesion: dict, clinical: dict, is_hu: bool) -> dict:
        out = {}
        bands = self.cfg.get_path("risk.risk_bands", {"low": 0.33, "high": 0.66})
        for t in PATIENT_TASKS:
            st = self.bank.statuses[t]
            if not is_hu:
                out[t] = self._unavailable(t, "Not applicable to non-HU (PNG/JPEG) input.")
                continue
            bundle = self.bank.tabular_model(t)
            if bundle is None:
                out[t] = self._unavailable(t, st.message)
                continue
            row = {**{k: v for k, v in lesion["radiomics"].items()}, **clinical}
            if lesion.get("deep_embedding"):
                row.update({f"deep_{i:03d}": v for i, v in enumerate(lesion["deep_embedding"])})
            cols = bundle["columns"]
            missing = [c for c in cols if c not in row and not c.startswith(("age", "sex", "smok", "pack"))]
            if missing:
                out[t] = self._unavailable(t, f"Required inputs unavailable ({len(missing)} features, e.g. "
                                              f"'{missing[0]}'); the malignancy CNN must be trained first.")
                continue
            X = np.array([[row.get(c, np.nan) for c in cols]], float)
            prob = bundle["model"].predict_proba(X)[0]
            probs = {c: round(float(p), 4) for c, p in zip(bundle["classes"], prob)}
            entry = {"task": TASKS[t].title, "state": st.state, "badge": st.badge, "probabilities": probs,
                     "wording": TASKS[t].output_wording, "caveat": TASKS[t].caveat,
                     "applies_to": f"index lesion #{lesion['id']} (assumes it is the primary NSCLC tumour)"}
            if len(prob) == 2:
                entry["risk_band"] = risk_band(float(prob[1]), bands["low"], bands["high"])
            out[t] = entry
        return out


def _resize_to(img: np.ndarray, shape) -> np.ndarray:
    from scipy import ndimage

    return ndimage.zoom(img, (shape[0] / img.shape[0], shape[1] / img.shape[1]), order=0)
