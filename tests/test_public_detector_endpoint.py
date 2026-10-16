import pytest

from api.argus_detect import evaluate_history


def test_demo_endpoint_runs_the_source_gradient_detector():
    assert evaluate_history("1, 1, 1, 11") == {
        "count": 4,
        "finding": {
            "type": "grad_explosion",
            "step": 4,
            "grad_norm": 11.0,
            "description": "grad norm 11.0 - exceeds threshold 10",
        },
    }
    assert evaluate_history("1, 1, 1") == {"count": 3, "finding": None}


@pytest.mark.parametrize("value", ["", "1, -1", "1, nan", "1, 1000001", ",".join(["1"] * 22)])
def test_demo_endpoint_rejects_unbounded_or_invalid_histories(value):
    with pytest.raises(ValueError):
        evaluate_history(value)
