import sys
from unittest.mock import MagicMock

import pytest

# Only mock pyarrow if it's not already available.
# This avoids breaking other tests in environments (like CI) where it is installed.
try:
    import pyarrow  # noqa: F401
    import pyarrow.parquet  # noqa: F401
except ImportError:
    mock_pa = MagicMock()
    mock_pq = MagicMock()
    sys.modules["pyarrow"] = mock_pa
    sys.modules["pyarrow.parquet"] = mock_pq

from alife_discovery.aggregation import (  # noqa: E402
    _mean,
    _percentile_pre_sorted,
    _to_float_list,
)


def test_percentile_pre_sorted_empty():
    assert _percentile_pre_sorted([], 0.5) is None


def test_percentile_pre_sorted_single():
    assert _percentile_pre_sorted([10.0], 0.5) == 10.0


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        ([10.0, 20.0], 0.0, 10.0),
        ([10.0, 20.0], 1.0, 20.0),
        ([10.0, 20.0], 0.5, 15.0),
        ([10.0, 20.0], 0.25, 12.5),
        ([10.0, 20.0, 30.0], 0.5, 20.0),
        ([10.0, 20.0, 30.0], 0.25, 15.0),
        ([10.0, 20.0, 30.0], 0.75, 25.0),
    ],
)
def test_percentile_pre_sorted_interpolation(
    values: list[float], q: float, expected: float
) -> None:
    assert _percentile_pre_sorted(values, q) == pytest.approx(expected)


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


def test_to_float_list_non_convertible_raises():
    rows = [{"val": "abc"}]
    with pytest.raises(ValueError):
        _to_float_list(rows, "val")


def test_mean_empty():
    assert _mean([]) is None


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 2.0, 3.0], 2.0),
        ([10.0], 10.0),
        ([-1.0, 1.0], 0.0),
    ],
)
def test_mean_values(values: list[float], expected: float) -> None:
    assert _mean(values) == pytest.approx(expected)
