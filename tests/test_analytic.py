import math
from dataclasses import replace

import pytest

from bsgreeks import EuropeanOption, bs_greeks, bs_price

# A grid covering ITM/ATM/OTM, short and long maturities, low and high vol, zero and negative rates.
GRID = [
    dict(spot=s, strike=100.0, maturity=t, rate=r, vol=v)
    for s in (70.0, 100.0, 130.0)
    for t in (0.1, 1.0, 5.0)
    for r in (-0.01, 0.0, 0.05)
    for v in (0.1, 0.4)
]


def both(params):
    return EuropeanOption(**params, kind="call"), EuropeanOption(**params, kind="put")


# --- External check: published textbook numbers --------------------------------------------------
# Hull, Options, Futures, and Other Derivatives. Hull prints these rounded, so the tolerance
# is half a unit in the last printed digit.


def test_hull_bsm_worked_example():
    # S=42, K=40, r=10%, vol=20%, six months: c = 4.76, p = 0.81.
    call, put = both(dict(spot=42.0, strike=40.0, maturity=0.5, rate=0.10, vol=0.20))
    assert bs_price(call) == pytest.approx(4.76, abs=0.005)
    assert bs_price(put) == pytest.approx(0.81, abs=0.005)


def test_hull_greek_letters_example():
    # Call on a non-dividend stock, S=49, K=50, r=5%, vol=20%, 20 weeks. Hull reports the
    # price as $2.40 and delta 0.522, gamma 0.066, vega 12.1, theta -4.31 per year
    # (-0.0118 per calendar day), rho 8.91.
    call = EuropeanOption(spot=49.0, strike=50.0, maturity=20 / 52, rate=0.05, vol=0.20)
    g = bs_greeks(call)
    assert bs_price(call) == pytest.approx(2.40, abs=0.005)
    assert g.delta == pytest.approx(0.522, abs=0.0005)
    assert g.gamma == pytest.approx(0.066, abs=0.0005)
    assert g.vega == pytest.approx(12.1, abs=0.05)
    assert g.theta == pytest.approx(-4.31, abs=0.005)
    assert g.theta / 365 == pytest.approx(-0.0118, abs=0.00005)
    assert g.rho == pytest.approx(8.91, abs=0.005)


# --- Internal consistency: relations any correct implementation must satisfy -----------------


@pytest.mark.parametrize("params", GRID)
def test_put_call_parity(params):
    call, put = both(params)
    forward = params["spot"] - params["strike"] * math.exp(-params["rate"] * params["maturity"])
    assert bs_price(call) - bs_price(put) == pytest.approx(forward, abs=1e-10)


@pytest.mark.parametrize("params", GRID)
def test_no_arbitrage_bounds(params):
    call, put = both(params)
    disc_k = params["strike"] * math.exp(-params["rate"] * params["maturity"])
    assert max(params["spot"] - disc_k, 0.0) - 1e-12 <= bs_price(call) <= params["spot"]
    assert max(disc_k - params["spot"], 0.0) - 1e-12 <= bs_price(put) <= disc_k


@pytest.mark.parametrize("params", GRID)
def test_greeks_match_finite_differences_of_price(params):
    field = {"delta": "spot", "vega": "vol", "rho": "rate"}
    for opt in both(params):
        g = bs_greeks(opt)
        for greek, name in field.items():
            h = 1e-5 * max(1.0, abs(getattr(opt, name)))
            up = bs_price(replace(opt, **{name: getattr(opt, name) + h}))
            dn = bs_price(replace(opt, **{name: getattr(opt, name) - h}))
            assert getattr(g, greek) == pytest.approx((up - dn) / (2 * h), rel=1e-5, abs=1e-6), greek

        h = 1e-3 * opt.spot
        up, mid, dn = (bs_price(replace(opt, spot=opt.spot + d)) for d in (h, 0.0, -h))
        assert g.gamma == pytest.approx((up - 2 * mid + dn) / h**2, rel=1e-4, abs=1e-6)

        h = 1e-5 * opt.maturity
        up = bs_price(replace(opt, maturity=opt.maturity + h))
        dn = bs_price(replace(opt, maturity=opt.maturity - h))
        assert g.theta == pytest.approx(-(up - dn) / (2 * h), rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("params", GRID)
def test_black_scholes_pde(params):
    # theta + r S delta + 1/2 vol^2 S^2 gamma = r V holds for every European payoff.
    for opt in both(params):
        g = bs_greeks(opt)
        lhs = g.theta + opt.rate * opt.spot * g.delta + 0.5 * opt.vol**2 * opt.spot**2 * g.gamma
        assert lhs == pytest.approx(opt.rate * bs_price(opt), abs=1e-10)


@pytest.mark.parametrize("params", GRID)
def test_call_put_greek_relations(params):
    gc, gp = (bs_greeks(o) for o in both(params))
    t, r, k = params["maturity"], params["rate"], params["strike"]
    assert gc.delta - gp.delta == pytest.approx(1.0, abs=1e-12)
    assert gc.gamma == pytest.approx(gp.gamma, rel=1e-12)
    assert gc.vega == pytest.approx(gp.vega, rel=1e-12)
    assert gc.rho - gp.rho == pytest.approx(k * t * math.exp(-r * t), rel=1e-12)


def test_zero_vol_limit_is_discounted_forward_payoff():
    opt = EuropeanOption(spot=110.0, strike=100.0, maturity=1.0, rate=0.05, vol=1e-8)
    assert bs_price(opt) == pytest.approx(110.0 - 100.0 * math.exp(-0.05), abs=1e-8)
    assert bs_price(replace(opt, kind="put")) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize(
    "bad",
    [dict(spot=0.0), dict(strike=-1.0), dict(maturity=0.0), dict(vol=0.0), dict(kind="straddle")],
)
def test_rejects_invalid_inputs(bad):
    params = dict(spot=100.0, strike=100.0, maturity=1.0, rate=0.01, vol=0.2) | bad
    with pytest.raises(ValueError):
        EuropeanOption(**params)
