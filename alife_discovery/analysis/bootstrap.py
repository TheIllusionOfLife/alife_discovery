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
    if n < 1:
        raise ValueError("n must be >= 1")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
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
    samples = rng.choice(excess_rates, size=(n_iter, n), replace=True)
    boot_means = samples.mean(axis=1)

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

    We use the same historical closed-form approximation used by this project:
    ``p_min ≈ (z_{1-α} + z_{power})^2 / (4n)``, where ``n = n_types``.
    This is a small-rate heuristic (not an exact binomial power calculation),
    retained for backward compatibility with prior reports.

    ``n_observations`` is accepted for interface compatibility; effective
    independent units are unique types (``n_types``).
    """
    del n_observations
    if n_types < 1:
        raise ValueError("n_types must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if not (0.0 < power < 1.0):
        raise ValueError("power must be in (0, 1)")
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    min_excess = ((z_alpha + z_beta) ** 2) / (4 * n_types)
    return float(min_excess)


def detection_power_simulated(
    *,
    n_types: int,
    true_excess_rate: float,
    alpha: float = 0.05,
    n_trials: int = 50_000,
    rng_seed: int = 0,
) -> float:
    """Monte Carlo power estimate for the same heuristic threshold.

    We simulate ``K ~ Binomial(n_types, true_excess_rate)`` and estimate
    ``P(K >= k_crit)`` with:

    ``k_crit = ceil(z_{1-α}^2 / 4)``.

    This critical count follows from the project's historical threshold
    ``p_hat >= z_{1-α}^2 / (4 n_types)``, hence the apparent cancellation of
    ``n_types`` in ``k_crit``. This function is therefore a consistency check
    for the legacy approximation, not a replacement for exact binomial power.
    """
    if n_types < 1:
        raise ValueError("n_types must be >= 1")
    if not (0.0 <= true_excess_rate <= 1.0):
        raise ValueError("true_excess_rate must be in [0, 1]")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    z_alpha = stats.norm.ppf(1 - alpha)
    k_crit = int(np.ceil((z_alpha**2) / 4.0))

    rng = np.random.default_rng(rng_seed)
    samples = rng.binomial(n=n_types, p=true_excess_rate, size=n_trials)
    return float(np.mean(samples >= k_crit))


def ks_pvalue_uniformity(pvalues: np.ndarray) -> tuple[float, float]:
    """KS test for p-value uniformity under the null.

    Tests whether the observed p-values follow a Uniform(0, 1) distribution.

    Returns:
        ``(ks_statistic, ks_pvalue)``. A high ks_pvalue means p-values
        are consistent with uniformity (null is well-calibrated).
    """
    result = stats.kstest(pvalues, "uniform")
    return (float(result.statistic), float(result.pvalue))
