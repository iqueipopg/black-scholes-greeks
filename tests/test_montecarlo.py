"""Monte Carlo against the closed form.

Every check is stated in standard errors estimated from the simulation itself, so a pass
means the two methods agree within the sampling error the Monte Carlo reports, not within
an arbitrary tolerance. Seeds are fixed: the tests are deterministic.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from bsgreeks import EuropeanOption, bs_greeks, bs_price, mc_price, mc_price_and_greeks, simulate_gbm_paths
from bsgreeks.montecarlo import GREEK_NAMES

HULL_BSM = EuropeanOption(spot=42.0, strike=40.0, maturity=0.5, rate=0.10, vol=0.20)
HULL_GREEKS = EuropeanOption(spot=49.0, strike=50.0, maturity=20 / 52, rate=0.05, vol=0.20)
OTHERS = [
    EuropeanOption(spot=100.0, strike=130.0, maturity=2.0, rate=0.03, vol=0.35),  # deep OTM call
    EuropeanOption(spot=100.0, strike=80.0, maturity=0.25, rate=0.0, vol=0.15),  # ITM call, zero rate
]
CASES = [replace(o, kind=k) for o in (HULL_BSM, HULL_GREEKS, *OTHERS) for k in ("call", "put")]
MAX_Z = 4.0  # |error| / SE; a correct estimator exceeds this about once in 16,000 checks


def ids(opt):
    return f"{opt.kind}-S{opt.spot:g}-K{opt.strike:g}-T{opt.maturity:.3g}"


@pytest.mark.parametrize("opt", CASES, ids=ids)
def test_price_and_greeks_agree_with_closed_form(opt):
    mc = mc_price_and_greeks(opt, n_paths=400_000, seed=7)
    exact = {"price": bs_price(opt), **vars(bs_greeks(opt))}
    for name in ("price", *GREEK_NAMES):
        assert abs(mc[name].z_score(exact[name])) < MAX_Z, (name, mc[name], exact[name])


@pytest.mark.parametrize("opt", [HULL_BSM, replace(HULL_GREEKS, kind="put")], ids=ids)
def test_multi_step_paths_give_the_same_price(opt):
    # Exact log-normal steps: 52 weekly steps must not introduce bias.
    est = mc_price(opt, n_paths=200_000, n_steps=52, seed=11)
    assert abs(est.z_score(bs_price(opt))) < MAX_Z


def test_confidence_interval_coverage():
    # The reported error margin is itself tested: across 400 independent small runs, the 95%
    # interval should contain the closed-form price about 95% of the time (binomial sd 1.1pp).
    exact = bs_price(HULL_GREEKS)
    hits = 0
    for seed in range(400):
        lo, hi = mc_price(HULL_GREEKS, n_paths=5_000, seed=seed).ci(0.95)
        hits += lo <= exact <= hi
    assert 0.92 <= hits / 400 <= 0.98


def test_standard_error_shrinks_as_one_over_sqrt_n():
    se = [mc_price(HULL_BSM, n_paths=n, seed=3).std_error for n in (25_000, 100_000, 400_000)]
    assert se[1] / se[0] == pytest.approx(0.5, rel=0.05)
    assert se[2] / se[1] == pytest.approx(0.5, rel=0.05)


def test_error_shrinks_with_more_paths():
    # Mean absolute error over 20 seeds falls roughly by half when n is multiplied by 4.
    exact = bs_price(HULL_BSM)
    mae = [np.mean([abs(mc_price(HULL_BSM, n, seed=s).value - exact) for s in range(20)]) for n in (10_000, 160_000)]
    assert mae[1] / mae[0] == pytest.approx(0.25, abs=0.12)


@pytest.mark.parametrize("opt", [HULL_BSM, replace(HULL_BSM, kind="put")], ids=ids)
def test_antithetic_is_unbiased_and_reduces_error(opt):
    plain = mc_price(opt, n_paths=200_000, seed=5)
    anti = mc_price(opt, n_paths=100_000, seed=5, antithetic=True)  # same number of paths
    assert abs(anti.z_score(bs_price(opt))) < MAX_Z
    assert anti.std_error < plain.std_error


def test_seed_reproducibility():
    a = mc_price_and_greeks(HULL_BSM, n_paths=10_000, seed=42)
    b = mc_price_and_greeks(HULL_BSM, n_paths=10_000, seed=42)
    assert a == b


def test_gbm_paths_are_risk_neutral_and_have_the_right_volatility():
    s0, r, vol, t, n_steps = 100.0, 0.04, 0.3, 1.0, 50
    paths = simulate_gbm_paths(s0, r, vol, t, n_steps, 100_000, np.random.default_rng(1))
    assert paths.shape == (100_000, n_steps + 1)
    assert np.all(paths[:, 0] == s0)
    assert np.all(paths > 0)

    # Discounted price is a martingale: E[exp(-rT) S_T] = S_0.
    disc_terminal = math.exp(-r * t) * paths[:, -1]
    se = disc_terminal.std(ddof=1) / math.sqrt(len(disc_terminal))
    assert abs(disc_terminal.mean() - s0) < MAX_Z * se

    # Log-returns over each step have variance vol^2 dt.
    log_ret = np.diff(np.log(paths), axis=1)
    assert log_ret.var(ddof=1) == pytest.approx(vol**2 * t / n_steps, rel=0.01)


def test_ci_width_matches_standard_error():
    est = mc_price(HULL_BSM, n_paths=10_000, seed=0)
    lo, hi = est.ci(0.95)
    assert (hi - lo) / 2 == pytest.approx(1.959964 * est.std_error, rel=1e-6)


def test_rejects_too_few_paths():
    with pytest.raises(ValueError):
        mc_price(HULL_BSM, n_paths=1)
