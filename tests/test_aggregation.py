import sys
from unittest.mock import MagicMock

# Mock pyarrow before importing aggregation
mock_pa = MagicMock()
mock_pq = MagicMock()
sys.modules["pyarrow"] = mock_pa
sys.modules["pyarrow.parquet"] = mock_pq

from objectless_alife.aggregation import (  # noqa: E402
    _mean,
    _percentile_pre_sorted,
    _to_float_list,
)


def test_percentile_pre_sorted_empty():
    assert _percentile_pre_sorted([], 0.5) is None


def test_percentile_pre_sorted_single():
    assert _percentile_pre_sorted([10.0], 0.5) == 10.0


def test_percentile_pre_sorted_two_elements():
    values = [10.0, 20.0]
    assert _percentile_pre_sorted(values, 0.0) == 10.0
    assert _percentile_pre_sorted(values, 1.0) == 20.0
    assert _percentile_pre_sorted(values, 0.5) == 15.0
    assert _percentile_pre_sorted(values, 0.25) == 12.5


def test_percentile_pre_sorted_multiple_elements():
    # Linear interpolation: pos = (n-1)*q
    # values: [10.0, 20.0, 30.0], n=3
    values = [10.0, 20.0, 30.0]
    # q=0.5 -> pos = (3-1)*0.5 = 1.0. lo=1, hi=1. fraction=0. 20.0
    assert _percentile_pre_sorted(values, 0.5) == 20.0
    # q=0.25 -> pos = (3-1)*0.25 = 0.5. lo=0, hi=1. fraction=0.5. 10*0.5 + 20*0.5 = 15.0
    assert _percentile_pre_sorted(values, 0.25) == 15.0
    # q=0.75 -> pos = (3-1)*0.75 = 1.5. lo=1, hi=2. fraction=0.5. 20*0.5 + 30*0.5 = 25.0
    assert _percentile_pre_sorted(values, 0.75) == 25.0


def test_percentile_pre_sorted_edge_q():
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile_pre_sorted(values, 0.0) == 1.0
    assert _percentile_pre_sorted(values, 1.0) == 4.0


def test_to_float_list_valid():
    rows = [{"a": 1.0}, {"a": 2.0}]
    assert _to_float_list(rows, "a") == [1.0, 2.0]


def test_to_float_list_mixed():
    rows = [
        {"val": 1.0},
        {"val": "2.5"},
        {"val": None},
        {"other": 5.0},
        {"val": float("nan")},
    ]
    # Note: _to_float_list uses float(value) so "2.5" should work.
    # numeric != numeric check skips NaN.
    result = _to_float_list(rows, "val")
    assert result == [1.0, 2.5]


def test_mean_empty():
    assert _mean([]) is None


def test_mean_values():
    assert _mean([1.0, 2.0, 3.0]) == 2.0
    assert _mean([10.0]) == 10.0
