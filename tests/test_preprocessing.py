import numpy as np
import pytest

from ctcancer.preprocessing import apply_window, extract_patch, preprocess_volume, resample_volume, slice_to_25d


def test_window_bounds():
    a = np.array([-2000, -1350, -600, 150, 3000], float)
    w = apply_window(a, -600, 1500)
    assert w.min() == 0 and w.max() == 1
    assert w[2] == pytest.approx(0.5)


def test_resample_to_isotropic():
    a = np.random.default_rng(0).normal(size=(10, 40, 40)).astype(np.float32)
    out, sp = resample_volume(a, (2.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    assert out.shape == (25, 20, 20)
    assert sp == pytest.approx((1.0, 1.0, 1.0))


def test_patch_padding_outside_volume():
    a = np.zeros((10, 10, 10), np.float32)
    p = extract_patch(a, (0, 0, 0), 8)
    assert p.shape == (8, 8, 8)
    assert (p[:4, :4, :4] == -1024).all() and (p[4:, 4:, 4:] == 0).all()


def test_25d_channels_are_adjacent_slices():
    v = np.arange(5)[:, None, None] * np.ones((5, 4, 4))
    x = slice_to_25d(v, 2)
    assert x.shape == (3, 4, 4) and list(x[:, 0, 0]) == [1, 2, 3]
    assert list(slice_to_25d(v, 0)[:, 0, 0]) == [0, 0, 1]  # edge clamping


def test_preprocess_volume(phantom):
    vol, _ = phantom
    pre = preprocess_volume(vol, {"hu_clip": [-1200, 600], "target_spacing_mm": [1.5, 1.5, 1.5],
                                  "lung_window": {"level": -600, "width": 1500}})
    assert pre.hu.max() <= 600 and pre.norm.min() >= 0 and pre.norm.max() <= 1
    assert pre.spacing == pytest.approx((1.5, 1.5, 1.5), abs=0.02)
    assert any("resampled" in s for s in pre.steps)
