"""Hierarchical testing utilities: Clopper-Pearson bounds, bootstrap CIs, power analysis.

These tools address the statistical independence concern (I4) raised by
peer reviewers: per-step entity observations are temporally dependent, so
naive per-observation testing overstates effective sample size.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound on binomial proportion.

    Returns the upper bound of a ``1 - alpha`` one-sided confidence interval
    for the true proportion, given ``k`` successes in ``n`` trials.

    Interpretation: "if excess exists, it is < returned_value with
    ``1 - alpha`` confidence."
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if k > n:
        raise ValueError("k must be <= n")
    if k == n:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def bootstrap_excess_ci(
    excess_rates: np.ndarray,
    n_iter: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """Block bootstrap 95% CI on overall excess rate.

    Each element of ``excess_rates`` is the excess rate for one run
    (proportion of entities with p < 0.05 in that run). Bootstrap
    resamples runs (not individual observations) to respect temporal
    dependence within runs.

    Returns:
        ``(lower, upper)`` bounds of the ``1 - alpha`` CI.
    """
    if len(excess_rates) == 0:
        return (0.0, 0.0)
    if len(excess_rates) == 1:
        val = float(excess_rates[0])
        return (val, val)

    rng = np.random.default_rng(rng_seed)
    n = len(excess_rates)
    boot_means = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sample = rng.choice(excess_rates, size=n, replace=True)
        boot_means[i] = sample.mean()

    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def detection_power(
    n_observations: int,
    n_types: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Minimum detectable excess rate at given power.

    Uses a normal approximation for the binomial test. Returns the
    smallest true excess rate that would be detected (rejected at
    significance ``alpha``) with probability ``power``, given
    ``n_observations`` independent tests across ``n_types`` unique types.

    The effective sample size is ``n_types`` (testing per unique type),
    not ``n_observations``.
    """
    n_eff = n_types
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    # Minimum detectable proportion difference from 0
    # Using normal approximation: p = ((z_alpha + z_beta) / sqrt(n_eff))^2 / 4
    # Simplified: p_min ≈ (z_alpha + z_beta)^2 / (4 * n_eff)
    min_excess = ((z_alpha + z_beta) ** 2) / (4 * n_eff)
    return float(min_excess)


def ks_pvalue_uniformity(pvalues: np.ndarray) -> tuple[float, float]:
    """KS test for p-value uniformity under the null.

    Tests whether the observed p-values follow a Uniform(0, 1) distribution.

    Returns:
        ``(ks_statistic, ks_pvalue)``. A high ks_pvalue means p-values
        are consistent with uniformity (null is well-calibrated).
    """
    result = stats.kstest(pvalues, "uniform")
    return (float(result.statistic), float(result.pvalue))
