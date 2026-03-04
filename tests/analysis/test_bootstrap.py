"""Tests for hierarchical testing, Clopper-Pearson bounds, and power analysis."""

from __future__ import annotations

import numpy as np
import pytest

from alife_discovery.analysis.bootstrap import (
    bootstrap_excess_ci,
    clopper_pearson_upper,
    detection_power,
    detection_power_simulated,
    ks_pvalue_uniformity,
)


class TestClopperPearsonUpper:
    def test_zero_successes(self) -> None:
        """k=0, n=100 → upper bound should be small but > 0."""
        ub = clopper_pearson_upper(k=0, n=100, alpha=0.05)
        assert 0.0 < ub < 0.05

    def test_all_successes(self) -> None:
        """k=n → upper bound is 1.0."""
        ub = clopper_pearson_upper(k=100, n=100, alpha=0.05)
        assert ub == 1.0

    def test_moderate_rate(self) -> None:
        """k=10, n=100 → upper bound > 0.10 (observed rate)."""
        ub = clopper_pearson_upper(k=10, n=100, alpha=0.05)
        assert ub > 0.10
        assert ub < 0.20

    def test_alpha_affects_width(self) -> None:
        """Smaller alpha → wider interval (higher upper bound)."""
        ub_95 = clopper_pearson_upper(k=5, n=100, alpha=0.05)
        ub_99 = clopper_pearson_upper(k=5, n=100, alpha=0.01)
        assert ub_99 > ub_95

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            clopper_pearson_upper(k=-1, n=100)

    def test_k_gt_n_raises(self) -> None:
        with pytest.raises(ValueError):
            clopper_pearson_upper(k=101, n=100)


class TestBootstrapExcessCI:
    def test_all_zeros(self) -> None:
        """All-zero excess rates → CI should be [0.0, 0.0]."""
        rates = np.zeros(100)
        lo, hi = bootstrap_excess_ci(rates, n_iter=1000, rng_seed=42)
        assert lo == 0.0
        assert hi == 0.0

    def test_nonzero_rates(self) -> None:
        """Mixture of rates → CI contains the mean."""
        rng = np.random.default_rng(42)
        rates = rng.uniform(0.0, 0.1, size=200)
        lo, hi = bootstrap_excess_ci(rates, n_iter=5000, rng_seed=42)
        assert lo <= rates.mean() <= hi
        assert lo >= 0.0
        assert hi <= 1.0

    def test_single_run(self) -> None:
        """Single run → degenerate CI = [rate, rate]."""
        rates = np.array([0.05])
        lo, hi = bootstrap_excess_ci(rates, n_iter=1000, rng_seed=0)
        assert lo == pytest.approx(0.05)
        assert hi == pytest.approx(0.05)


class TestDetectionPower:
    def test_large_sample_high_power(self) -> None:
        """Large sample → can detect small excess."""
        min_excess = detection_power(n_observations=100_000, n_types=281, alpha=0.05)
        assert min_excess < 0.01  # can detect < 1% excess

    def test_small_sample_lower_power(self) -> None:
        """Small sample → needs larger excess to detect."""
        min_excess = detection_power(n_observations=100, n_types=10, alpha=0.05)
        assert min_excess > 0.01

    def test_returns_positive(self) -> None:
        min_excess = detection_power(n_observations=1000, n_types=50, alpha=0.05)
        assert min_excess > 0.0

    def test_simulated_power_matches_target_approximately(self) -> None:
        min_excess = detection_power(n_observations=100_000, n_types=281, alpha=0.05)
        simulated = detection_power_simulated(
            n_types=281,
            true_excess_rate=min_excess,
            alpha=0.05,
            n_trials=20_000,
            rng_seed=42,
        )
        assert 0.72 <= simulated <= 0.88


class TestKsPvalueUniformity:
    def test_uniform_pvalues(self) -> None:
        """Uniform p-values → high KS p-value (fail to reject H0)."""
        rng = np.random.default_rng(42)
        pvalues = rng.uniform(0, 1, size=1000)
        ks_stat, ks_pval = ks_pvalue_uniformity(pvalues)
        assert ks_pval > 0.05

    def test_non_uniform_pvalues(self) -> None:
        """All p-values near 0 → very low KS p-value."""
        pvalues = np.full(100, 0.001)
        ks_stat, ks_pval = ks_pvalue_uniformity(pvalues)
        assert ks_pval < 0.01
