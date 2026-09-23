# black-scholes-greeks

[![tests](https://github.com/iqueipopg/black-scholes-greeks/actions/workflows/tests.yml/badge.svg)](https://github.com/iqueipopg/black-scholes-greeks/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

**A European option pricer with two independent engines that are made to check each other.**
The Black-Scholes closed form gives the price and the five Greeks (delta, gamma,
vega, theta, rho). A Monte Carlo simulation of geometric Brownian motion paths
gives the same six numbers from scratch, with its own standard errors. The two
share nothing but the option definition, so when they agree within the error the
simulation reports, that agreement means something. On top of that, the closed form
is pinned to published numbers from Hull, so the pair is not only consistent with
itself but with an external reference.

Small on purpose: about 330 lines of library code, numpy only, 297 tests that run
in two seconds.

## Result

Hull's worked example (S = 42, K = 40, T = 0.5, r = 10%, vol = 20%), one million
Monte Carlo samples, seed 2026:

| | closed form | Monte Carlo | std. error | error | error / SE |
|---|---|---|---|---|---|
| call price | **4.7594** | **4.7580** | 0.0050 | -0.0014 | -0.28 |
| put price | 0.8086 | 0.8080 | 0.0018 | -0.0006 | -0.36 |

Hull prints 4.76 and 0.81. The Monte Carlo 95% interval for the call,
[4.7483, 4.7678], contains the closed-form price. The error falls as the Monte
Carlo error bar says it should:

| samples | MC call price | std. error | error vs. closed form |
|---|---|---|---|
| 10,000 | 4.7789 | 0.0495 | +0.0195 |
| 100,000 | 4.7542 | 0.0156 | -0.0052 |
| 1,000,000 | 4.7580 | 0.0050 | -0.0014 |
| 10,000,000 | 4.7594 | 0.0016 | -0.0001 |

The Greeks, on Hull's running example for the Greek letters chapter (S = 49,
K = 50, T = 20 weeks, r = 5%, vol = 20%), one million samples:

| | closed form | Hull (rounded) | Monte Carlo | std. error | error / SE |
|---|---|---|---|---|---|
| price | 2.4005 | 2.40 | 2.3999 | 0.0038 | -0.16 |
| delta | 0.5216 | 0.522 | 0.5211 | 0.0006 | -0.85 |
| gamma | 0.0655 | 0.066 | 0.0661 | 0.0004 | +1.31 |
| vega | 12.1055 | 12.1 | 12.1049 | 0.0206 | -0.03 |
| theta (per year) | -4.3053 | -4.31 | -4.3041 | 0.0062 | +0.21 |
| rho | 8.9070 | 8.91 | 8.8984 | 0.0094 | -0.91 |

Every Monte Carlo figure lies within 1.3 standard errors of the closed form.
Reproduce both tables with `python -m bsgreeks`.

## The model

Under the risk-neutral measure the stock follows
$dS_t = r S_t dt + \sigma S_t dW_t$. For a European option with strike $K$ and
maturity $T$,

$$
d_1 = \frac{\ln(S/K) + (r + \sigma^2/2) T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}
$$

$$
c = S N(d_1) - K e^{-rT} N(d_2), \qquad p = K e^{-rT} N(-d_2) - S N(-d_1)
$$

| Greek | call | put |
|---|---|---|
| delta, $\partial V/\partial S$ | $N(d_1)$ | $N(d_1) - 1$ |
| gamma, $\partial^2 V/\partial S^2$ | $\varphi(d_1) / (S\sigma\sqrt{T})$ | same |
| vega, $\partial V/\partial\sigma$ | $S \varphi(d_1)\sqrt{T}$ | same |
| theta, $-\partial V/\partial T$ | $-\frac{S\varphi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2)$ | $-\frac{S\varphi(d_1)\sigma}{2\sqrt{T}} + rKe^{-rT}N(-d_2)$ |
| rho, $\partial V/\partial r$ | $KTe^{-rT}N(d_2)$ | $-KTe^{-rT}N(-d_2)$ |

Units follow Hull: vega and rho per unit (divide by 100 for one percentage point),
theta per year (divide by 365 for one calendar day).

## How each piece is validated

The design separates what is exact from what is estimated, and checks each
against something it does not depend on.

**1. Closed form against a published source.** The prices and Greeks above are
compared with the numbers printed in Hull, *Options, Futures, and Other
Derivatives*: the worked example of the Black-Scholes-Merton chapter (c = 4.76,
p = 0.81) and the running example of the Greek letters chapter (price 2.40,
delta 0.522, gamma 0.066, vega 12.1, theta -4.31 per year or -0.0118 per day,
rho 8.91). Tolerance is half a unit in the last digit Hull prints.

**2. Closed form against itself, on a 54-point grid** (in/at/out of the money,
maturities from five weeks to five years, zero and negative rates):

- put-call parity, $c - p = S - Ke^{-rT}$, to $10^{-10}$;
- no-arbitrage bounds on both prices;
- each Greek equals the finite-difference derivative of the pricing function;
- the Black-Scholes PDE, $\Theta + rS\Delta + \tfrac12\sigma^2S^2\Gamma = rV$,
  holds to $10^{-10}$, which ties theta, delta and gamma together;
- call-put relations: $\Delta_c - \Delta_p = 1$, equal gamma and vega,
  $\rho_c - \rho_p = KTe^{-rT}$.

**3. Monte Carlo against the closed form.** The simulation draws GBM paths with
the exact log-normal step
$S_{t+\Delta t} = S_t \exp((r - \sigma^2/2)\Delta t + \sigma\sqrt{\Delta t} Z)$,
so the number of steps adds no discretisation bias and the only error is sampling
error. It never calls the normal CDF or $d_1$. Greeks come from the same draws:
pathwise derivatives for delta, vega, theta and rho, and the likelihood-ratio
estimator for gamma, since the pathwise method breaks on the kink of the payoff
(Glasserman 2003, ch. 7). The tests then check:

- price and all five Greeks within 4 standard errors of the closed form, for
  calls and puts on four parameter sets including a deep out-of-the-money call
  and a zero rate;
- the same price with 52 weekly steps per path as with one step;
- **the error bar itself**: over 400 independent runs the 95% interval must
  contain the closed-form price between 92% and 98% of the time. (Over 4,000
  runs it does 94.5% of the time.) A pricer whose standard errors were off by a
  factor of 1.5 in either direction fails this test, even though its prices
  would look fine;
- the standard error shrinks as $1/\sqrt{n}$, and the actual error with it;
- antithetic variates stay unbiased and lower the standard error at equal path count;
- simulated paths are risk-neutral ($E[e^{-rT}S_T] = S_0$) with per-step
  log-return variance $\sigma^2\Delta t$.

All seeds are fixed, so the suite is deterministic. As a check on the checks,
the tests were run against deliberately broken versions of the code (wrong sign
in $d_1$, $N(d_1)$ in place of $N(d_2)$ in rho, wrong sign in the Monte Carlo
theta, a missing term in the pathwise vega, standard errors scaled by 0.5 or
1.5). Every one of them fails the suite.

## Limitations

This is the textbook model and inherits all of its assumptions. None of them is
hidden:

- **Constant volatility.** Real option prices imply a volatility that varies with
  strike and maturity (smile and skew). Black-Scholes cannot produce it, and no
  implied-volatility or surface fitting is included here.
- **No dividends.** A dividend-paying stock needs the continuous-yield (Merton)
  adjustment or explicit cash dividends; neither is implemented.
- **Constant, known interest rate**, one rate for borrowing and lending.
- **Log-normal prices, continuous paths.** No jumps, no fat tails beyond what a
  log-normal gives.
- **Frictionless markets**: continuous hedging, no transaction costs, no
  short-sale limits. The Greeks are the hedge ratios of that idealised world.
- **European exercise only.** The Monte Carlo engine here prices a terminal payoff;
  it does not handle early exercise (American options) or path-dependent payoffs.
- **The Monte Carlo check validates the implementation, not the model.** Both
  engines assume GBM, so their agreement says the formulas are coded correctly,
  not that GBM describes the market.

## Usage

```bash
pip install -r requirements.txt
python -m bsgreeks                                   # the two Hull cases
python -m bsgreeks --spot 100 --strike 105 --maturity 1 --rate 0.03 --vol 0.25 --antithetic
pytest -q
```

```python
from bsgreeks import EuropeanOption, bs_price, bs_greeks, mc_price_and_greeks

opt = EuropeanOption(spot=42, strike=40, maturity=0.5, rate=0.10, vol=0.20, kind="call")
bs_price(opt)                                   # 4.7594
bs_greeks(opt).delta                            # 0.7791
mc = mc_price_and_greeks(opt, n_paths=1_000_000, seed=2026)
mc["price"].value, mc["price"].std_error        # (4.7580, 0.0050)
mc["price"].ci(0.95)                            # (4.7483, 4.7678)
```

## Layout

```
bsgreeks/analytic.py     closed-form price and Greeks
bsgreeks/montecarlo.py   GBM path simulation, MC price and Greeks with standard errors
bsgreeks/__main__.py     closed form vs Monte Carlo report
tests/                   textbook values, analytic identities, MC agreement and calibration
```

## References

- Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Merton, R. C. (1973). Theory of Rational Option Pricing. *Bell Journal of Economics and Management Science*, 4(1), 141-183.
- Hull, J. C. *Options, Futures, and Other Derivatives*. Pearson.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
