import json

import numpy as np
import pytest

from noise_injector.corruptions.annotation import apply_annotation_op
from noise_injector.corruptions.qa import apply_qa_op
from noise_injector.spec import ANNOTATION_FAMILIES, QA_FAMILIES


def make_koi(n=10):
    return {
        f"<c{i},CAM_FRONT,1.0,1.0>": {
            "Category": "Vehicle" if i % 2 == 0 else "Pedestrian",
            "Status": "Moving" if i % 3 == 0 else "Stationary",
            "Visual_description": f"obj {i}",
            "2d_bbox": [float(i * 20), 10.0, float(i * 20 + 40), 50.0],
        }
        for i in range(n)
    }


def make_qa(n_per_cat=6):
    return {
        "perception": [{"Q": f"Q{i} <c1,CAM_FRONT,1.0,1.0>", "A": f"answer {i}", "C": None} for i in range(n_per_cat)],
        "planning": [{"Q": f"Plan{i}", "A": "Decelerate gradually without braking.", "C": None} for i in range(n_per_cat)],
        "behavior": [{"Q": "Behavior?", "A": "The ego vehicle is going straight.", "C": None}],
    }


@pytest.mark.parametrize("fam", sorted(ANNOTATION_FAMILIES))
def test_annotation_ops_execute(fam):
    rng = np.random.default_rng(0)
    params = ANNOTATION_FAMILIES[fam].params(5)
    koi = make_koi()
    detail = apply_annotation_op(fam, koi, params, rng)
    assert detail is not None
    if fam == "max_det_truncation":
        assert len(koi) <= params["cap"]
    if fam == "missed_detections":
        assert len(koi) < 10
    if fam == "phantom_objects":
        assert any(k.startswith("<p") for k in koi)


def test_bbox_jitter_preserves_pairs(tmp_path):
    koi = {"<c1,CAM_FRONT,1.0,1.0>": {"Category": "Vehicle", "Status": None, "Visual_description": "",
                                       "2d_bbox": [[10.0, 10.0], [40.0, 40.0]]}}
    apply_annotation_op("bbox_jitter", koi, {"sigma_px": 3.0}, np.random.default_rng(1))
    b = koi["<c1,CAM_FRONT,1.0,1.0>"]["2d_bbox"]
    assert isinstance(b[0], list) and len(b[0]) == 2


def test_status_flip_only_known_states():
    koi = {
        "a": {"Category": "Vehicle", "Status": "Moving", "2d_bbox": [0, 0, 10, 10]},
        "b": {"Category": "Vehicle", "Status": None, "2d_bbox": [0, 0, 10, 10]},
    }
    apply_annotation_op("status_flip", koi, {"rate": 1.0}, np.random.default_rng(2))
    assert koi["a"]["Status"] == "Stationary"
    assert koi["b"]["Status"] is None


@pytest.mark.parametrize("fam", sorted(QA_FAMILIES))
def test_qa_ops_execute(fam):
    rng = np.random.default_rng(0)
    qa = make_qa()
    params = {"rate": 1.0}
    detail = apply_qa_op(fam, qa, params, rng)
    assert detail is not None
    if fam == "qa_drop":
        assert qa["perception"] == []
    if fam == "qa_duplicate":
        assert len(qa["behavior"]) == 2
    if fam == "qa_crossref_break":
        assert "<c18" in qa["perception"][0]["Q"]


def test_qa_wrong_answer_changes_answers():
    qa = make_qa()
    before = [i["A"] for i in qa["perception"]]
    apply_qa_op("qa_wrong_answer", qa, {"rate": 1.0}, np.random.default_rng(3))
    after = [i["A"] for i in qa["perception"]]
    assert after != before


def test_qa_answer_drift_semantic():
    qa = {"planning": [{"Q": "Plan", "A": "turn left at the intersection", "C": None}]}
    apply_qa_op("qa_answer_drift", qa, {"rate": 1.0}, np.random.default_rng(4))
    assert qa["planning"][0]["A"] == "turn right at the intersection"
