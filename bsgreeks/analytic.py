"""Closed-form Black-Scholes prices and Greeks for European options.

Conventions: continuously compounded rate, annualised volatility, maturity in
years, no dividends. Vega and rho are per unit (1.00 = 100 percentage points),
theta is per year of calendar time. Divide theta by 365 for a per-day figure
and vega or rho by 100 for a per-point figure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _SQRT_2)


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


@dataclass(frozen=True)
class EuropeanOption:
    spot: float
    strike: float
    maturity: float
    rate: float
    vol: float
    kind: str = "call"

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")
        for name in ("spot", "strike", "maturity", "vol"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")

    @property
    def is_call(self) -> bool:
        return self.kind == "call"


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def d1_d2(opt: EuropeanOption) -> tuple[float, float]:
    vol_sqrt_t = opt.vol * math.sqrt(opt.maturity)
    d1 = (math.log(opt.spot / opt.strike) + (opt.rate + 0.5 * opt.vol**2) * opt.maturity) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(opt: EuropeanOption) -> float:
    d1, d2 = d1_d2(opt)
    disc_k = opt.strike * math.exp(-opt.rate * opt.maturity)
    if opt.is_call:
        return opt.spot * norm_cdf(d1) - disc_k * norm_cdf(d2)
    return disc_k * norm_cdf(-d2) - opt.spot * norm_cdf(-d1)


def bs_greeks(opt: EuropeanOption) -> Greeks:
    d1, d2 = d1_d2(opt)
    s, k, t, r, vol = opt.spot, opt.strike, opt.maturity, opt.rate, opt.vol
    sqrt_t = math.sqrt(t)
    disc_k = k * math.exp(-r * t)
    pdf_d1 = norm_pdf(d1)

    # Gamma and vega are the same for calls and puts: the difference between
    # the two is a forward, which is linear in S and does not depend on vol.
    gamma = pdf_d1 / (s * vol * sqrt_t)
    vega = s * pdf_d1 * sqrt_t
    time_decay = -s * pdf_d1 * vol / (2.0 * sqrt_t)

    if opt.is_call:
        delta = norm_cdf(d1)
        theta = time_decay - r * disc_k * norm_cdf(d2)
        rho = t * disc_k * norm_cdf(d2)
    else:
        delta = norm_cdf(d1) - 1.0
        theta = time_decay + r * disc_k * norm_cdf(-d2)
        rho = -t * disc_k * norm_cdf(-d2)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
