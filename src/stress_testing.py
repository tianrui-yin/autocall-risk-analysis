"""
Stress testing for autocallable products.

Historical and hypothetical scenarios measuring impact on product value.

Reference: Basel III stress testing framework; FRTB guidelines.
"""

import numpy as np
import pandas as pd
from .autocall import Autocallable


HISTORICAL_SCENARIOS = {
    "Lehman 2008": {"spot_shock": -0.35, "vol_shock": 0.30},
    "COVID Mars 2020": {"spot_shock": -0.30, "vol_shock": 0.40},
    "Crise Euro 2011": {"spot_shock": -0.20, "vol_shock": 0.15},
    "Flash Crash 2010": {"spot_shock": -0.10, "vol_shock": 0.20},
    "Correction Fev 2018": {"spot_shock": -0.12, "vol_shock": 0.25},
    "SG Kerviel 2008": {"spot_shock": -0.07, "vol_shock": 0.10},
}

HYPOTHETICAL_SCENARIOS = {
    "Marche stable (+5%)": {"spot_shock": 0.05, "vol_shock": -0.02},
    "Correction moderee (-10%)": {"spot_shock": -0.10, "vol_shock": 0.10},
    "Crash severe (-25%)": {"spot_shock": -0.25, "vol_shock": 0.25},
    "Crash extreme (-40%)": {"spot_shock": -0.40, "vol_shock": 0.35},
    "Vol spike seul": {"spot_shock": 0.0, "vol_shock": 0.20},
    "Vol collapse": {"spot_shock": 0.05, "vol_shock": -0.10},
}


def stress_test(
    product: Autocallable,
    spot_shock: float,
    vol_shock: float,
    n_paths: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    Apply a stress scenario and measure impact.

    Parameters
    ----------
    spot_shock : float
        Relative spot change (e.g., -0.30 = -30%).
    vol_shock : float
        Absolute vol change (e.g., 0.20 = +20 vol points).

    Returns
    -------
    dict with base price, stressed price, P&L impact, and risk metrics.
    """
    base = product.price(n_paths=n_paths, seed=seed)

    new_S0 = product.S0 * (1 + spot_shock)
    new_sigma = max(product.sigma + vol_shock, 0.01)

    stressed = product.price_for_greeks(
        S0=new_S0, sigma=new_sigma, n_paths=n_paths, seed=seed
    )

    vol_model = getattr(product, "vol_model", None)
    if vol_model is not None and hasattr(vol_model, "shift"):
        vol_model = vol_model.shift(vol_shock)
    stressed_full = Autocallable(
        S0=new_S0,
        autocall_barrier=product.autocall_barrier / new_S0,
        coupon_barrier=product.coupon_barrier / new_S0,
        ki_barrier=product.ki_barrier / new_S0,
        coupon_rate=product.coupon_rate,
        n_observations=product.n_observations,
        T=product.T,
        r=product.r,
        sigma=new_sigma,
        notional=product.notional,
        vol_model=vol_model,
    )
    stressed_detail = stressed_full.price(n_paths=n_paths, seed=seed)

    pnl_impact = stressed - base["price"]
    pnl_pct = pnl_impact / product.notional * 100

    return {
        "base_price": base["price"],
        "stressed_price": stressed,
        "pnl_impact": pnl_impact,
        "pnl_pct": pnl_pct,
        "ki_prob_base": base["ki_probability"],
        "ki_prob_stressed": stressed_detail["ki_probability"],
        "autocall_prob_base": base["total_autocall_prob"],
        "autocall_prob_stressed": stressed_detail["total_autocall_prob"],
    }


def run_all_scenarios(
    product: Autocallable,
    n_paths: int = 100_000,
    seed: int = 42,
) -> dict:
    """Run all historical and hypothetical stress scenarios."""
    results = {}

    for name, params in HISTORICAL_SCENARIOS.items():
        results[name] = stress_test(product, **params, n_paths=n_paths, seed=seed)
        results[name]["type"] = "historical"

    for name, params in HYPOTHETICAL_SCENARIOS.items():
        results[name] = stress_test(product, **params, n_paths=n_paths, seed=seed)
        results[name]["type"] = "hypothetical"

    return results


def find_worst_scenario(results: dict) -> tuple[str, dict]:
    """Identify the worst-case scenario by P&L impact."""
    worst_name = min(results, key=lambda k: results[k]["pnl_impact"])
    return worst_name, results[worst_name]


def stress_grid(
    product: Autocallable,
    spot_shocks: list[float] | None = None,
    vol_shocks: list[float] | None = None,
    n_paths: int = 100_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a 2D stress grid: spot changes x vol changes.

    For each (spot_shock, vol_shock) pair, reprices the product and collects
    price, P&L, KI probability, and autocall probability.

    Parameters
    ----------
    product : Autocallable
        The product to stress test.
    spot_shocks : list of float, optional
        Relative spot shocks (e.g., -0.30 = -30%).
        Default: [-30%, -20%, -10%, -5%, 0%, +5%, +10%, +20%].
    vol_shocks : list of float, optional
        Relative vol shocks as multipliers of current sigma
        (e.g., -0.50 = halve vol, +1.0 = double vol).
        Default: [-50%, -20%, 0%, +20%, +50%, +100%].
    n_paths : int
        Monte Carlo paths for each repricing.
    seed : int
        Random seed.

    Returns
    -------
    (DataFrame, pivot_table)
        DataFrame with columns: spot_shock, vol_shock, price, pnl, pnl_pct,
            ki_prob, autocall_prob.
        Pivot table: spot_shock (rows) x vol_shock (cols) -> pnl.
    """
    if spot_shocks is None:
        spot_shocks = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
    if vol_shocks is None:
        vol_shocks = [-0.50, -0.20, 0.0, 0.20, 0.50, 1.00]

    # Base price
    base_result = product.price(n_paths=n_paths, seed=seed)
    base_price = base_result["price"]

    rows = []
    for s_shock in spot_shocks:
        for v_shock in vol_shocks:
            new_S0 = product.S0 * (1 + s_shock)
            new_sigma = max(product.sigma * (1 + v_shock), 0.01)

            # Quick reprice for price/pnl
            stressed_price = product.price_for_greeks(
                S0=new_S0, sigma=new_sigma, n_paths=n_paths, seed=seed,
            )

            # Full reprice for detailed metrics (ki_prob, autocall_prob)
            vol_model = getattr(product, "vol_model", None)
            if vol_model is not None and hasattr(vol_model, "shift"):
                sigma_shift = new_sigma - product.sigma
                vol_model = vol_model.shift(sigma_shift)

            stressed_full = Autocallable(
                S0=new_S0,
                autocall_barrier=product.autocall_barrier / new_S0,
                coupon_barrier=product.coupon_barrier / new_S0,
                ki_barrier=product.ki_barrier / new_S0,
                coupon_rate=product.coupon_rate,
                n_observations=product.n_observations,
                T=product.T,
                r=product.r,
                sigma=new_sigma,
                notional=product.notional,
                vol_model=vol_model,
            )
            detail = stressed_full.price(n_paths=n_paths, seed=seed)

            pnl = stressed_price - base_price
            pnl_pct = pnl / product.notional * 100

            rows.append({
                "spot_shock": s_shock,
                "vol_shock": v_shock,
                "price": stressed_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "ki_prob": detail["ki_probability"],
                "autocall_prob": detail["total_autocall_prob"],
            })

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        values="pnl", index="spot_shock", columns="vol_shock", aggfunc="first",
    )

    return df, pivot


def sensitivity_surface(
    product: Autocallable,
    param1_name: str,
    param1_range: list[float],
    param2_name: str,
    param2_range: list[float],
    n_paths: int = 100_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generic 2-parameter sensitivity surface.

    Reprices the product for each combination of (param1, param2),
    where parameters are set as absolute values (not shocks).

    Parameters
    ----------
    product : Autocallable
        The product to analyze.
    param1_name : str
        Name of the first parameter (e.g., 'S0', 'sigma', 'r', 'T').
    param1_range : list of float
        Absolute values for the first parameter.
    param2_name : str
        Name of the second parameter.
    param2_range : list of float
        Absolute values for the second parameter.
    n_paths : int
        Monte Carlo paths for each repricing.
    seed : int
        Random seed.

    Returns
    -------
    (DataFrame, pivot_table)
        DataFrame with columns: param1_name, param2_name, price.
        Pivot table: param1 (rows) x param2 (cols) -> price.
    """
    rows = []
    for p1_val in param1_range:
        for p2_val in param2_range:
            kwargs = {param1_name: p1_val, param2_name: p2_val}
            price = product.price_for_greeks(n_paths=n_paths, seed=seed, **kwargs)
            rows.append({
                param1_name: p1_val,
                param2_name: p2_val,
                "price": price,
            })

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        values="price", index=param1_name, columns=param2_name, aggfunc="first",
    )

    return df, pivot
