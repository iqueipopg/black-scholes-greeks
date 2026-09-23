"""Closed form vs Monte Carlo, side by side.

python -m bsgreeks                      # Hull's two textbook cases
python -m bsgreeks --spot 100 --strike 105 --maturity 1 --rate 0.03 --vol 0.25
"""

from __future__ import annotations

import argparse

from .analytic import EuropeanOption, bs_greeks, bs_price
from .montecarlo import GREEK_NAMES, mc_price_and_greeks

# Hull, Options, Futures, and Other Derivatives: the worked example of the
# Black-Scholes-Merton chapter and the running example of the Greek letters chapter.
HULL_CASES = {
    "Hull BSM example (S=42, K=40, T=0.5, r=10%, vol=20%)": dict(
        spot=42.0, strike=40.0, maturity=0.5, rate=0.10, vol=0.20
    ),
    "Hull Greeks example (S=49, K=50, T=20/52, r=5%, vol=20%)": dict(
        spot=49.0, strike=50.0, maturity=20 / 52, rate=0.05, vol=0.20
    ),
}


def report(opt: EuropeanOption, n_paths: int, n_steps: int, seed: int, antithetic: bool) -> None:
    mc = mc_price_and_greeks(opt, n_paths, n_steps=n_steps, seed=seed, antithetic=antithetic)
    exact = {"price": bs_price(opt), **vars(bs_greeks(opt))}
    print(f"  {opt.kind:<4} {'closed form':>12} {'Monte Carlo':>12} {'std err':>10} {'error':>10} {'err/SE':>7}")
    for name in ("price", *GREEK_NAMES):
        est = mc[name]
        err = est.value - exact[name]
        print(
            f"  {name:<6}{exact[name]:>10.4f} {est.value:>12.4f} {est.std_error:>10.4f} {err:>+10.4f}"
            f" {est.z_score(exact[name]):>+7.2f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m bsgreeks", description=__doc__.splitlines()[0])
    for name in ("spot", "strike", "maturity", "rate", "vol"):
        p.add_argument(f"--{name}", type=float)
    p.add_argument("--paths", type=int, default=1_000_000)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--antithetic", action="store_true")
    a = p.parse_args()

    params = dict(spot=a.spot, strike=a.strike, maturity=a.maturity, rate=a.rate, vol=a.vol)
    if all(v is None for v in params.values()):
        cases = HULL_CASES
    elif any(v is None for v in params.values()):
        p.error("give all of --spot --strike --maturity --rate --vol, or none")
    else:
        cases = {"custom": params}

    print(f"Monte Carlo: {a.paths:,} samples, {a.steps} step(s) per path, seed {a.seed}, antithetic={a.antithetic}")
    for title, params in cases.items():
        print(f"\n{title}")
        for kind in ("call", "put"):
            report(EuropeanOption(**params, kind=kind), a.paths, a.steps, a.seed, a.antithetic)


if __name__ == "__main__":
    main()
