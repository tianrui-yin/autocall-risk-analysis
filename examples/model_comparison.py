"""
Model Comparison: Constant Vol vs Term Structure vs Local Vol vs Heston

Prices the same Phoenix Autocallable under four volatility models to
quantify model risk for barrier products.

Key insight: constant vol underestimates knock-in probability because it
ignores the vol smile (in reality, vol rises as spot drops — the "leverage
effect"). Local vol and Heston capture this through different mechanisms,
giving a more realistic risk picture.

Usage:
    python examples/model_comparison.py
"""

import sys
import os
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.autocall import Autocallable
from src.local_vol import ConstantVol, TermStructureVol, build_local_vol_from_market
from src.vol_surface import build_vol_surface_from_market
from src.heston import HestonModel
from src.market_data import load_sample_data

N_PATHS = 100_000
SEED = 42


def main():
    print("=" * 78)
    print("MODEL COMPARISON: CONSTANT VOL vs TERM STRUCTURE vs LOCAL VOL vs HESTON")
    print("=" * 78)

    # ── 1. Load market data ──
    print("\n1. Loading market data...")
    data = load_sample_data()
    print(f"   EUROSTOXX 50 spot: {data['spot_stoxx']:.2f}")
    print(f"   SPY spot:          {data['spot_spy']:.2f}")
    print(f"   Risk-free rate:    {data['risk_free_rate']:.2%}")
    print(f"   Realized vol (1y): {data['realized_vol']:.2%}")

    # ── 2. Build vol surface ──
    print("\n2. Building implied volatility surface from SPY options...")
    surface = build_vol_surface_from_market(use_sample=True)
    print(f"   Calibrated {surface.n_slices} maturity slices")
    for T in surface.maturities:
        print(f"   T={T:.3f}y: ATM vol = {surface.atm_vol(T):.2%}")

    # ── 3. Build vol models ──
    print("\n3. Constructing volatility models...")

    spot = data["spot_spy"]
    r = data["risk_free_rate"]

    # Maturity for the autocallable (must be within calibrated range)
    T_product = min(0.8, float(surface.maturities[-1]) - 0.01)

    atm_vol = surface.atm_vol(T_product)
    print(f"   ATM vol at T={T_product:.2f}y: {atm_vol:.2%}")

    # Model 1: Constant vol (current project baseline)
    const_vol = ConstantVol(atm_vol)
    print(f"   Model 1: {const_vol.model_name}")

    # Model 2: ATM term structure
    ts_vol = TermStructureVol.from_surface(surface)
    print(f"   Model 2: {ts_vol.model_name}")

    # Model 3: Local vol (Dupire)
    from src.local_vol import LocalVol
    lv = LocalVol(surface, spot_ref=spot, r=r)
    lv.precompute_grid()
    print(f"   Model 3: {lv.model_name}")

    # Model 4: Heston stochastic volatility
    # Typical equity parameters: negative rho (leverage effect),
    # v0 = ATM_vol^2, theta = long-term variance
    heston = HestonModel(
        kappa=2.0,
        theta=atm_vol**2,
        xi=0.3,
        v0=atm_vol**2,
        rho=-0.7,
    )
    print(f"   Model 4: {heston.model_name}")
    print(f"            Feller condition: {'satisfied' if heston.feller_satisfied else 'VIOLATED'}")

    # ── 4. Define product ──
    print(f"\n4. Product: Phoenix Autocallable on SPY")
    product_params = dict(
        S0=spot,
        autocall_barrier=1.0,
        coupon_barrier=0.80,
        ki_barrier=0.60,
        coupon_rate=0.05,
        n_observations=4,
        T=T_product,
        r=r,
        sigma=atm_vol,
        notional=100.0,
    )

    ac_base = Autocallable(**product_params)
    print(f"   {ac_base.description()}")
    print(f"   Observation dates: {ac_base.observation_times}")

    # ── 5. Price under each model ──
    print(f"\n5. Pricing with {N_PATHS:,} Monte Carlo paths...")

    models = {
        "Constant Vol": None,  # Uses sigma parameter directly
        "Term Structure": ts_vol,
        "Local Vol (Dupire)": lv,
        "Heston (SV)": heston,
    }

    results = {}
    for name, vol_model in models.items():
        ac = Autocallable(**product_params, vol_model=vol_model)
        result = ac.price(n_paths=N_PATHS, n_steps_per_period=25, seed=SEED)
        results[name] = result
        print(f"   {name:25s}: price={result['price']:.2f}  "
              f"AC={result['total_autocall_prob']:.1%}  "
              f"KI={result['ki_probability']:.1%}  "
              f"E[life]={result['expected_life']:.2f}y")

    # ── 6. Comparison table ──
    print(f"\n{'─' * 72}")
    print("6. MODEL COMPARISON TABLE")
    print(f"{'─' * 72}")
    header = f"{'Model':25s} {'Price':>8s} {'KI Prob':>8s} {'AC Prob':>8s} {'E[Life]':>8s}"
    print(header)
    print("─" * len(header))

    for name, r in results.items():
        print(f"{name:25s} {r['price']:>8.2f} {r['ki_probability']:>7.1%} "
              f"{r['total_autocall_prob']:>7.1%} {r['expected_life']:>7.2f}y")

    # ── 7. Key insights ──
    r_const = results["Constant Vol"]
    r_lv = results["Local Vol (Dupire)"]
    r_heston = results["Heston (SV)"]

    ki_diff_lv = r_lv["ki_probability"] - r_const["ki_probability"]
    ki_diff_h = r_heston["ki_probability"] - r_const["ki_probability"]
    price_diff_lv = r_lv["price"] - r_const["price"]
    price_diff_h = r_heston["price"] - r_const["price"]

    print(f"\n{'─' * 72}")
    print("7. KEY INSIGHTS (Model Risk)")
    print(f"{'─' * 72}")
    print(f"""
   Local Vol vs Constant Vol:
   - KI probability:  {r_const['ki_probability']:.1%} → {r_lv['ki_probability']:.1%}  (delta = {ki_diff_lv:+.1%})
   - Price difference: {price_diff_lv:+.2f} (on notional 100)

   Heston vs Constant Vol:
   - KI probability:  {r_const['ki_probability']:.1%} → {r_heston['ki_probability']:.1%}  (delta = {ki_diff_h:+.1%})
   - Price difference: {price_diff_h:+.2f} (on notional 100)

   WHY: Constant vol assumes volatility stays fixed regardless of spot moves.
   Local vol captures the vol smile deterministically (vol = f(S, t)).
   Heston adds stochastic volatility — vol itself is random and correlated
   with spot (rho < 0). Both models capture the leverage effect:
     - Higher vol on the downside → higher probability of breaching the KI barrier
     - This means constant vol UNDERESTIMATES the knock-in risk
     - Heston also captures vol-of-vol (fat tails), which further impacts
       barrier products

   Local Vol vs Heston: Local vol is calibrated to match the full implied vol
   smile by construction. Heston generates an approximate smile via its 5
   parameters. The difference reveals model risk — the same market data
   produces different exotic prices depending on the model choice.

   IMPLICATION: A desk using constant vol for hedging would be underhedged
   against downside scenarios — exactly when hedging matters most.
    """)

    # ── 8. Delta comparison ──
    print(f"{'─' * 78}")
    print("8. DELTA PROFILE COMPARISON")
    print(f"{'─' * 78}")
    from src import greeks as greeks_module

    spots = np.array([0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1]) * spot
    print(f"   {'S/S0':>6s}  {'Delta(Const)':>14s}  {'Delta(LocalVol)':>16s}  "
          f"{'Delta(Heston)':>14s}  {'LV-Const':>10s}  {'H-Const':>10s}")
    print(f"   {'─' * 78}")

    for s in spots:
        ac_c = Autocallable(**product_params)
        ac_c.S0 = s
        d_const = greeks_module.delta(ac_c, n_paths=50_000, seed=SEED)
        ac_c.S0 = spot

        ac_l = Autocallable(**product_params, vol_model=lv)
        ac_l.S0 = s
        d_lv = greeks_module.delta(ac_l, n_paths=50_000, seed=SEED)
        ac_l.S0 = spot

        ac_h = Autocallable(**product_params, vol_model=heston)
        ac_h.S0 = s
        d_h = greeks_module.delta(ac_h, n_paths=50_000, seed=SEED)
        ac_h.S0 = spot

        print(f"   {s/spot:>6.0%}  {d_const:>+14.4f}  {d_lv:>+16.4f}  "
              f"{d_h:>+14.4f}  {d_lv-d_const:>+10.4f}  {d_h-d_const:>+10.4f}")

    print(f"\n{'=' * 78}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
