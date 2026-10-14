import numpy as np
import pytest

from noise_injector.spec import ALL_FAMILIES, HARNESS_OPS, PATTERNS, family, list_families
from noise_injector.temporal import activation_mask, derive_seed


def test_family_lookup():
    assert family("fog").layer == "A"
    assert family("qa_drop").layer == "C"
    assert family("vocab_drop").layer == "F"
    with pytest.raises(KeyError):
        family("nonexistent")


def test_severity_clamping():
    for s in (0, 6, -3):
        expected = family("fog").params(max(1, min(5, s)))["amount"]
        assert family("fog").params(s)["amount"] == expected


def test_temporal_persistent_and_transient():
    rng = np.random.default_rng(0)
    m = activation_mask("persistent", 5, rng)
    assert m["frames"] == [True] * 5
    m = activation_mask("transient", 5, rng)
    assert sum(m["frames"]) == 1


def test_temporal_burst_contiguous():
    rng = np.random.default_rng(1)
    for _ in range(20):
        m = activation_mask("burst", 8, rng, severity=3)
        idxs = [i for i, b in enumerate(m["frames"]) if b]
        assert idxs == list(range(idxs[0], idxs[0] + len(idxs)))


def test_temporal_camera_scoped():
    rng = np.random.default_rng(2)
    m = activation_mask("camera_scoped", 4, rng, n_cameras=6)
    assert m["frames"] == [True] * 4
    assert 0 <= m["camera"] < 6


def test_derive_seed_stable():
    assert derive_seed(42, "a", "b") == derive_seed(42, "a", "b")
    assert derive_seed(42, "a", "b") != derive_seed(42, "a", "c")


def test_list_families_layers():
    layers = list_families()
    assert set(layers) >= {"A", "B", "C", "F"}
    assert all("family" in e for entries in layers.values() for e in entries)
