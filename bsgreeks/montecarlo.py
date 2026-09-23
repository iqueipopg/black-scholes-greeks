"""Monte Carlo pricer for European options under geometric Brownian motion.

Shares nothing with ``analytic`` except the option definition: no normal CDF,
no d1/d2. Paths are simulated under the risk-neutral measure with the exact
log-normal step,

    S(t + dt) = S(t) * exp((r - vol^2 / 2) dt + vol sqrt(dt) Z),

so there is no discretisation bias for any number of steps, and the only error
left is sampling error, which is estimated from the same draws.

Greeks come from the same draws as the price: pathwise derivatives for delta,
vega, theta and rho, and the likelihood-ratio estimator for gamma, where the
pathwise method fails because the payoff has a kink at the strike
(Glasserman 2003, ch. 7).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .analytic import EuropeanOption

GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")

# Cap on normals held in memory at once (n_paths_in_batch * n_steps).
_MAX_BATCH_DRAWS = 4_000_000


@dataclass(frozen=True)
class MCEstimate:
    value: float
    std_error: float
    n_samples: int

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        return self.value - z * self.std_error, self.value + z * self.std_error

    def z_score(self, reference: float) -> float:
        """Distance to a reference value in standard errors."""
        return (self.value - reference) / self.std_error


def simulate_gbm_paths(
    spot: float,
    rate: float,
    vol: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Risk-neutral GBM paths, shape (n_paths, n_steps + 1), first column = spot."""
    dt = maturity / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_steps = (rate - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)], axis=1)
    return spot * np.exp(log_paths)


def _terminal_batches(
    opt: EuropeanOption, n_paths: int, n_steps: int, rng: np.random.Generator, antithetic: bool
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (Z, S_T) batches, where Z = W_T / sqrt(T) is the standardised
    terminal Brownian value of each simulated path.

    Each path is built from n_steps independent increments. With antithetic
    sampling every batch has shape (2, m): row 1 is the mirror path (-Z) of
    row 0, and the pair counts as one sample.
    """
    batch = max(1, _MAX_BATCH_DRAWS // n_steps)
    drift = (opt.rate - 0.5 * opt.vol**2) * opt.maturity
    done = 0
    while done < n_paths:
        m = min(batch, n_paths - done)
        increments = rng.standard_normal((m, n_steps))
        z = increments.sum(axis=1) / math.sqrt(n_steps)
        if antithetic:
            z = np.stack([z, -z])
        s_t = opt.spot * np.exp(drift + opt.vol * math.sqrt(opt.maturity) * z)
        yield z, s_t
        done += m


def _sample_estimators(opt: EuropeanOption, z: np.ndarray, s_t: np.ndarray) -> dict[str, np.ndarray]:
    s, k, t, r, vol = opt.spot, opt.strike, opt.maturity, opt.rate, opt.vol
    sqrt_t = math.sqrt(t)
    disc = math.exp(-r * t)

    if opt.is_call:
        payoff = np.maximum(s_t - k, 0.0)
        dpayoff = (s_t > k).astype(float)  # derivative of the payoff in S_T
    else:
        payoff = np.maximum(k - s_t, 0.0)
        dpayoff = -(s_t < k).astype(float)

    # dS_T/dx for each parameter x, holding Z fixed.
    ds_dspot = s_t / s
    ds_dvol = s_t * (sqrt_t * z - vol * t)
    ds_drate = s_t * t
    ds_dmat = s_t * (r - 0.5 * vol**2 + vol * z / (2.0 * sqrt_t))

    # Likelihood-ratio weight for the second derivative of the density of S_T in spot.
    lr_gamma = ((z**2 - 1.0) / (vol**2 * t) - z / (vol * sqrt_t)) / s**2

    return {
        "price": disc * payoff,
        "delta": disc * dpayoff * ds_dspot,
        "gamma": disc * payoff * lr_gamma,
        "vega": disc * dpayoff * ds_dvol,
        "rho": disc * (dpayoff * ds_drate - t * payoff),
        # theta = -dV/dT
        "theta": -disc * (dpayoff * ds_dmat - r * payoff),
    }


def mc_price_and_greeks(
    opt: EuropeanOption,
    n_paths: int,
    n_steps: int = 1,
    seed: int | None = None,
    antithetic: bool = False,
) -> dict[str, MCEstimate]:
    """Price and all five Greeks from one set of simulated paths.

    ``n_paths`` counts independent samples; with ``antithetic=True`` each sample
    is a pair of mirror paths, so 2 * n_paths paths are simulated. Standard
    errors are computed over the independent samples.
    """
    if n_paths < 2 or n_steps < 1:
        raise ValueError("need n_paths >= 2 and n_steps >= 1")
    rng = np.random.default_rng(seed)
    sums: dict[str, float] = {}
    sums_sq: dict[str, float] = {}
    for z, s_t in _terminal_batches(opt, n_paths, n_steps, rng, antithetic):
        for name, x in _sample_estimators(opt, z, s_t).items():
            if antithetic:
                x = x.mean(axis=0)
            sums[name] = sums.get(name, 0.0) + float(x.sum())
            sums_sq[name] = sums_sq.get(name, 0.0) + float((x * x).sum())

    out = {}
    for name, total in sums.items():
        mean = total / n_paths
        var = max(sums_sq[name] / n_paths - mean**2, 0.0) * n_paths / (n_paths - 1)
        out[name] = MCEstimate(value=mean, std_error=math.sqrt(var / n_paths), n_samples=n_paths)
    return out


def mc_price(
    opt: EuropeanOption,
    n_paths: int,
    n_steps: int = 1,
    seed: int | None = None,
    antithetic: bool = False,
) -> MCEstimate:
    return mc_price_and_greeks(opt, n_paths, n_steps, seed, antithetic)["price"]
