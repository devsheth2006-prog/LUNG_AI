import numpy as np
import pytest

from ctcancer.detection import generate_candidates
from ctcancer.evaluation.metrics import dice, hausdorff95, iou, segmentation_metrics
from ctcancer.features import describe_location, extract_radiomics, shape_features
from ctcancer.segmentation import classical_lung_mask, split_left_right


def _sphere(n=48, r=10):
    z, y, x = np.mgrid[:n, :n, :n] - n // 2
    return (z ** 2 + y ** 2 + x ** 2) <= r ** 2


def test_classical_lung_mask_on_phantom(phantom):
    vol, nod = phantom
    lung = classical_lung_mask(vol.array, vol.spacing)
    lr = split_left_right(lung)
    assert set(np.unique(lr)) == {0, 1, 2}
    # closing/hole-filling re-includes the solid nodule inside the lung
    assert lung[nod].mean() > 0.9
    # patient right lung (label 1) lies at lower x (LPS)
    assert np.argwhere(lr == 1)[:, 2].mean() < np.argwhere(lr == 2)[:, 2].mean()


def test_candidate_detection_finds_phantom_nodule(phantom):
    vol, nod = phantom
    lung = classical_lung_mask(vol.array, vol.spacing)
    cands = generate_candidates(vol.array, lung, vol.spacing)
    assert len(cands) >= 1
    c = np.argwhere(nod).mean(0)
    best = min(cands, key=lambda k: np.linalg.norm((np.array(k.centroid_zyx) - c) * vol.spacing))
    assert np.linalg.norm((np.array(best.centroid_zyx) - c) * vol.spacing) < 3.0
    assert best.equiv_diameter_mm == pytest.approx(14.0, abs=2.5)


def test_shape_features_sphere_accuracy():
    m = _sphere(r=10)
    f = shape_features(m, (1, 1, 1))
    assert f["volume_mm3"] == pytest.approx(4 / 3 * np.pi * 1000, rel=0.05)
    assert f["equiv_diameter_mm"] == pytest.approx(20, rel=0.03)
    assert f["max_axial_diameter_mm"] == pytest.approx(20, abs=1.5)
    assert f["sphericity"] > 0.85
    assert f["elongation"] == pytest.approx(1, abs=0.05)


def test_radiomics_keys_and_intensity():
    m = _sphere()
    hu = np.where(m, 30.0, -850.0) + np.random.default_rng(0).normal(0, 5, m.shape)
    f = extract_radiomics(hu, m, (1, 1, 1))
    assert {"fo_mean", "glcm_contrast", "glcm_homogeneity", "sphericity"} <= set(f)
    assert f["fo_mean"] == pytest.approx(30, abs=2)
    assert f["fo_frac_solid"] == pytest.approx(1.0)


def test_location_description(phantom):
    vol, nod = phantom
    lung = classical_lung_mask(vol.array, vol.spacing)
    loc = describe_location(np.argwhere(nod).mean(0), split_left_right(lung), lung)
    assert loc["side"] == "right lung"
    assert "lobe not determined" in loc["text"]


def test_segmentation_metrics_known_values():
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    a[2:6, 2:6] = True
    b[2:6, 4:8] = True
    assert dice(a, b) == pytest.approx(0.5)
    assert iou(a, b) == pytest.approx(1 / 3)
    assert dice(a, a) == 1.0 and hausdorff95(a, a) == 0.0
    m = segmentation_metrics(a, b)
    assert m["sensitivity"] == pytest.approx(0.5)
