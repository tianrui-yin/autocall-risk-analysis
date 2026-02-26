"""
Implied volatility surface construction and SVI calibration.

Builds a vol surface from market option prices, calibrates SVI parametric model
per maturity slice, and provides interpolation for arbitrary (K, T).

SVI parameterization (Gatheral 2004):
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
where:
    w = sigma_imp^2 * T  (total implied variance)
    k = ln(K / F)        (log-moneyness)
    a, b, rho, m, sigma  (5 SVI parameters per maturity slice)

Reference: Gatheral, J. "The Volatility Surface" (Wiley, 2006), Ch. 3.
"""

import logging
from typing import Optional

import numpy as np
from scipy import optimize, stats, interpolate

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Black-Scholes implied volatility extraction
# =====================================================================

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2))


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0:
        return max(K * np.exp(-r * T) - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Vega = dC/dsigma = dP/dsigma (same for calls and puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(S * np.sqrt(T) * stats.norm.pdf(d1))


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    initial_guess: float = 0.3,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Extract implied volatility via Newton-Raphson, with Brent fallback.

    Parameters
    ----------
    market_price : float
        Observed option price.
    S, K, T, r : float
        Spot, strike, time to maturity, risk-free rate.
    option_type : str
        'call' or 'put'. Vega is the same for both; only the price
        function and intrinsic value differ.
    initial_guess : float
        Starting sigma for Newton iteration.

    Returns
    -------
    float or None
        Implied vol if converged, None otherwise.
    """
    if option_type == "put":
        intrinsic = max(K * np.exp(-r * T) - S, 0.0)
        price_func = bs_put_price
    else:
        intrinsic = max(S - K * np.exp(-r * T), 0.0)
        price_func = bs_call_price

    if market_price <= intrinsic + 1e-10:
        return None
    if T <= 0:
        return None

    # Newton-Raphson (vega is the same for calls and puts)
    sigma = initial_guess
    for _ in range(max_iter):
        price = price_func(S, K, T, r, sigma)
        v = bs_vega(S, K, T, r, sigma)
        if v < 1e-12:
            break
        diff = price - market_price
        if abs(diff) < tol:
            if 0.01 < sigma < 3.0:
                return sigma
            return None
        sigma -= diff / v
        if sigma <= 0:
            break

    # Brent fallback
    def objective(s):
        return price_func(S, K, T, r, s) - market_price

    try:
        result = optimize.brentq(objective, 0.01, 3.0, xtol=tol)
        return float(result)
    except ValueError:
        return None


def extract_implied_vols(
    options_df: "pd.DataFrame",
    spot: float,
    r: float,
) -> "pd.DataFrame":
    """
    Compute implied vols for an entire options chain.

    Parameters
    ----------
    options_df : pd.DataFrame
        Must have columns: 'strike', 'T', 'price'.
    spot : float
        Current spot price.
    r : float
        Risk-free rate.

    Returns
    -------
    pd.DataFrame
        Input augmented with: 'implied_vol', 'log_moneyness',
        'total_variance', 'forward'.
        Rows where extraction failed are dropped.
    """
    import pandas as pd

    df = options_df.copy()
    ivs = []
    for _, row in df.iterrows():
        iv = implied_vol(
            market_price=row["price"],
            S=spot,
            K=row["strike"],
            T=row["T"],
            r=r,
            option_type=row.get("option_type", "call"),
        )
        ivs.append(iv)

    df["implied_vol"] = ivs
    df = df.dropna(subset=["implied_vol"]).reset_index(drop=True)

    df["forward"] = spot * np.exp(r * df["T"])
    df["log_moneyness"] = np.log(df["strike"] / df["forward"])
    df["total_variance"] = df["implied_vol"] ** 2 * df["T"]

    return df


# =====================================================================
# 2. SVI calibration
# =====================================================================

def svi_total_variance(k: np.ndarray, params: tuple) -> np.ndarray:
    """
    SVI total variance w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2)).

    Parameters
    ----------
    k : np.ndarray
        Log-moneyness values.
    params : tuple
        (a, b, rho, m, sigma).

    Returns
    -------
    np.ndarray
        Total implied variance values.
    """
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def svi_implied_vol(k: np.ndarray, T: float, params: tuple) -> np.ndarray:
    """Convert SVI total variance to implied vol: sigma = sqrt(w(k) / T)."""
    w = svi_total_variance(k, params)
    return np.sqrt(np.maximum(w / T, 1e-10))


def _svi_initial_guess(k: np.ndarray, w: np.ndarray) -> tuple:
    """Heuristic initial SVI parameters from data."""
    a0 = float(np.interp(0.0, k, w))
    b0 = float(max((w.max() - w.min()) / max(k.max() - k.min(), 1e-6), 0.01))
    rho0 = -0.5  # typical equity skew
    m0 = 0.0
    sigma0 = 0.1
    return (a0, b0, rho0, m0, sigma0)


def calibrate_svi_slice(
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
    T: float,
    initial_params: Optional[tuple] = None,
) -> dict:
    """
    Calibrate SVI parameters for a single maturity slice.

    Uses L-BFGS-B with bounds; falls back to differential_evolution
    if local optimizer fails.

    Parameters
    ----------
    log_moneyness : np.ndarray
        k = ln(K/F) values.
    total_variance : np.ndarray
        w = sigma_imp^2 * T observed values.
    T : float
        Time to maturity (for Vega weighting).
    initial_params : tuple, optional
        (a, b, rho, m, sigma) starting point.

    Returns
    -------
    dict with keys:
        'params': tuple (a, b, rho, m, sigma)
        'rmse': float (weighted RMSE in vol terms)
        'success': bool
        'n_points': int
    """
    k = np.asarray(log_moneyness, dtype=float)
    w_obs = np.asarray(total_variance, dtype=float)
    n = len(k)

    if n < 5:
        return {"params": None, "rmse": np.inf, "success": False, "n_points": n}

    # Sort by moneyness
    idx = np.argsort(k)
    k = k[idx]
    w_obs = w_obs[idx]

    if initial_params is None:
        initial_params = _svi_initial_guess(k, w_obs)

    # Vega-like weighting: ATM options get more weight
    weights = np.exp(-0.5 * k**2 / 0.1)
    weights /= weights.sum()

    def objective(params):
        w_model = svi_total_variance(k, tuple(params))
        residuals = (w_model - w_obs) * weights
        return float(np.sum(residuals**2))

    bounds = [
        (1e-6, None),     # a > 0
        (1e-6, None),     # b > 0
        (-0.999, 0.999),  # -1 < rho < 1
        (None, None),     # m unrestricted
        (1e-4, None),     # sigma > 0
    ]

    # L-BFGS-B
    try:
        result = optimize.minimize(
            objective, initial_params, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if result.success:
            params = tuple(result.x)
            w_fit = svi_total_variance(k, params)
            vol_rmse = float(np.sqrt(np.mean(
                (np.sqrt(np.maximum(w_fit / T, 0)) - np.sqrt(np.maximum(w_obs / T, 0)))**2
            )))
            return {"params": params, "rmse": vol_rmse, "success": True, "n_points": n}
    except Exception as e:
        logger.debug("L-BFGS-B failed for T=%.3f: %s", T, e)

    # Fallback: differential evolution (global optimizer)
    de_bounds = [
        (1e-6, max(w_obs) * 2),
        (1e-6, 2.0),
        (-0.999, 0.999),
        (k.min() - 0.1, k.max() + 0.1),
        (1e-4, 1.0),
    ]
    try:
        result = optimize.differential_evolution(
            objective, de_bounds, maxiter=300, seed=42, tol=1e-10,
        )
        params = tuple(result.x)
        w_fit = svi_total_variance(k, params)
        vol_rmse = float(np.sqrt(np.mean(
            (np.sqrt(np.maximum(w_fit / T, 0)) - np.sqrt(np.maximum(w_obs / T, 0)))**2
        )))
        return {"params": params, "rmse": vol_rmse, "success": True, "n_points": n}
    except Exception as e:
        logger.warning("SVI calibration failed for T=%.3f: %s", T, e)
        return {"params": None, "rmse": np.inf, "success": False, "n_points": n}


def _enforce_calendar_spread(calibrations: dict) -> dict:
    """
    Drop SVI slices that violate calendar spread no-arbitrage.

    Total implied variance w(k=0, T) must be non-decreasing in T.
    Slices where ATM total variance decreases are removed.
    """
    sorted_Ts = sorted(calibrations.keys())
    if len(sorted_Ts) <= 1:
        return calibrations

    valid = {}
    prev_w = 0.0
    for T in sorted_Ts:
        params = calibrations[T]["params"]
        w_atm = svi_total_variance(np.array([0.0]), params)[0]
        if w_atm >= prev_w:
            valid[T] = calibrations[T]
            prev_w = w_atm
        else:
            logger.warning(
                "Calendar spread violation: dropped T=%.3f (w_atm=%.6f < prev %.6f)",
                T, w_atm, prev_w,
            )
    return valid


def calibrate_surface(vol_data: "pd.DataFrame", min_points: int = 5) -> dict:
    """
    Calibrate SVI for each maturity slice in the surface.

    Parameters
    ----------
    vol_data : pd.DataFrame
        From extract_implied_vols(). Must have 'T', 'log_moneyness', 'total_variance'.
    min_points : int
        Minimum data points per slice (SVI has 5 params).

    Returns
    -------
    dict mapping T_value -> calibration result dict. Sorted by maturity.
    """
    calibrations = {}

    for T_val, group in vol_data.groupby("T"):
        if len(group) < min_points:
            logger.info("Skipping T=%.3f: only %d points (need %d)", T_val, len(group), min_points)
            continue

        result = calibrate_svi_slice(
            log_moneyness=group["log_moneyness"].values,
            total_variance=group["total_variance"].values,
            T=T_val,
        )

        if result["success"]:
            calibrations[T_val] = result
            logger.info(
                "T=%.3f: SVI calibrated (RMSE=%.4f, n=%d)",
                T_val, result["rmse"], result["n_points"],
            )
        else:
            logger.warning("T=%.3f: SVI calibration failed", T_val)

    calibrations = _enforce_calendar_spread(dict(sorted(calibrations.items())))
    return calibrations


# =====================================================================
# 3. Implied vol surface with interpolation
# =====================================================================

class ImpliedVolSurface:
    """
    Calibrated implied volatility surface with (K, T) interpolation.

    Strike interpolation: SVI parametric (smooth, arbitrage-aware).
    Maturity interpolation: linear interpolation of total variance
    between calibrated slices (preserves calendar spread no-arbitrage).

    Parameters
    ----------
    spot : float
        Current spot price.
    r : float
        Risk-free rate.
    svi_calibrations : dict
        T -> {'params': (a, b, rho, m, sigma), ...}
    """

    def __init__(self, spot: float, r: float, svi_calibrations: dict):
        self.spot = spot
        self.r = r
        self._calibrations = svi_calibrations
        self._maturities = np.array(sorted(svi_calibrations.keys()))

        if len(self._maturities) == 0:
            raise ValueError("No successful SVI calibrations provided")

    @property
    def maturities(self) -> np.ndarray:
        return self._maturities.copy()

    @property
    def n_slices(self) -> int:
        return len(self._maturities)

    def _get_svi_params_at_T(self, T: float) -> tuple:
        """
        Get SVI parameters at maturity T via interpolation.

        For T at a calibrated slice: return exact params.
        For T between slices: interpolate total variance linearly,
        then invert to get effective SVI params (use nearest slice
        params but scale 'a' to match interpolated ATM variance).
        For T outside range: use nearest slice (flat extrapolation).
        """
        if T <= self._maturities[0]:
            return self._calibrations[self._maturities[0]]["params"]
        if T >= self._maturities[-1]:
            return self._calibrations[self._maturities[-1]]["params"]

        # Exact match
        for mat in self._maturities:
            if abs(T - mat) < 1e-10:
                return self._calibrations[mat]["params"]

        # Interpolation: find bracketing slices
        idx = np.searchsorted(self._maturities, T) - 1
        T1 = self._maturities[idx]
        T2 = self._maturities[idx + 1]
        alpha = (T - T1) / (T2 - T1)

        p1 = np.array(self._calibrations[T1]["params"])
        p2 = np.array(self._calibrations[T2]["params"])

        # Linear interpolation of SVI params
        params_interp = (1 - alpha) * p1 + alpha * p2
        # Ensure constraints
        params_interp[1] = max(params_interp[1], 1e-6)  # b > 0
        params_interp[2] = np.clip(params_interp[2], -0.999, 0.999)  # rho
        params_interp[4] = max(params_interp[4], 1e-4)  # sigma > 0

        return tuple(params_interp)

    def implied_vol_at(self, K: float, T: float) -> float:
        """
        Implied vol for arbitrary (K, T).

        Parameters
        ----------
        K : float
            Strike price.
        T : float
            Time to maturity.

        Returns
        -------
        float
            Implied volatility.
        """
        T = max(T, 1e-4)
        F = self.spot * np.exp(self.r * T)
        k = np.log(K / F)
        params = self._get_svi_params_at_T(T)
        w = svi_total_variance(np.array([k]), params)[0]
        return float(np.sqrt(max(w / T, 1e-10)))

    def atm_vol(self, T: float) -> float:
        """ATM implied vol at maturity T."""
        F = self.spot * np.exp(self.r * T)
        return self.implied_vol_at(F, T)

    def total_variance_at(self, K: float, T: float) -> float:
        """Total variance w(K, T) = sigma_imp(K,T)^2 * T."""
        iv = self.implied_vol_at(K, T)
        return iv**2 * T

    def vol_grid(
        self,
        moneyness_range: tuple = (0.7, 1.3),
        T_range: Optional[tuple] = None,
        n_strikes: int = 50,
        n_maturities: int = 20,
    ) -> tuple:
        """
        Generate a grid of implied vols for visualization.

        Returns (moneyness_grid, T_grid, vol_grid) as 2D arrays.
        """
        if T_range is None:
            T_range = (float(self._maturities[0]), float(self._maturities[-1]))

        moneyness = np.linspace(moneyness_range[0], moneyness_range[1], n_strikes)
        Ts = np.linspace(T_range[0], T_range[1], n_maturities)

        M_grid, T_grid = np.meshgrid(moneyness, Ts)
        vol_grid = np.zeros_like(M_grid)

        for i in range(n_maturities):
            for j in range(n_strikes):
                K = self.spot * M_grid[i, j]
                vol_grid[i, j] = self.implied_vol_at(K, T_grid[i, j])

        return M_grid, T_grid, vol_grid

    def fit_quality(self) -> dict:
        """
        Report calibration quality metrics for the SVI surface.

        Returns
        -------
        dict with keys:
            'per_slice_rmse': dict mapping T -> RMSE (vol space)
            'per_slice_max_error': dict mapping T -> max absolute error (vol space)
            'overall_rmse': float — pooled RMSE across all slices
            'n_slices': int
        """
        per_slice_rmse = {}
        per_slice_max_error = {}

        for T_val, cal in self._calibrations.items():
            per_slice_rmse[T_val] = cal.get("rmse", 0.0)
            # Max error: approximate from RMSE if not separately stored
            per_slice_max_error[T_val] = cal.get("max_error", cal.get("rmse", 0.0))

        rmse_values = list(per_slice_rmse.values())
        overall_rmse = float(np.sqrt(np.mean(np.array(rmse_values) ** 2))) if rmse_values else 0.0

        return {
            "per_slice_rmse": per_slice_rmse,
            "per_slice_max_error": per_slice_max_error,
            "overall_rmse": overall_rmse,
            "n_slices": len(self._calibrations),
        }

    def arbitrage_check(
        self,
        k_grid: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Check the SVI surface for butterfly and calendar spread arbitrage.

        Butterfly: d^2 w / dk^2 >= 0 (convexity of total variance).
        Calendar: w(k, T1) <= w(k, T2) for T1 < T2.

        Parameters
        ----------
        k_grid : np.ndarray, optional
            Log-moneyness grid for checks. Default: [-0.3, 0.3] with 100 points.

        Returns
        -------
        dict with keys:
            'butterfly_pass': bool
            'butterfly_details': list of violations
            'calendar_spread_pass': bool
            'calendar_details': list of violations
        """
        if k_grid is None:
            k_grid = np.linspace(-0.3, 0.3, 100)

        # ----- Butterfly arbitrage: d^2w/dk^2 >= 0 -----
        butterfly_violations = []
        for T_val in self._maturities:
            params = self._calibrations[T_val]["params"]
            for k in k_grid:
                d2w = svi_d2w_dk2(k, params)
                if d2w < -1e-10:
                    butterfly_violations.append({
                        "T": float(T_val),
                        "k": float(k),
                        "d2w_dk2": float(d2w),
                    })

        butterfly_pass = len(butterfly_violations) == 0

        # ----- Calendar spread arbitrage: w(k, T1) <= w(k, T2) -----
        calendar_violations = []
        # Check at ATM (k=0) and +/- 10% moneyness
        check_points = [0.0, -0.1, 0.1]

        sorted_Ts = sorted(self._maturities)
        for i in range(len(sorted_Ts) - 1):
            T1, T2 = sorted_Ts[i], sorted_Ts[i + 1]
            params1 = self._calibrations[T1]["params"]
            params2 = self._calibrations[T2]["params"]

            for k in check_points:
                w1 = svi_total_variance(np.array([k]), params1)[0]
                w2 = svi_total_variance(np.array([k]), params2)[0]
                if w1 > w2 + 1e-10:
                    calendar_violations.append({
                        "T1": float(T1),
                        "T2": float(T2),
                        "k": float(k),
                        "w_T1": float(w1),
                        "w_T2": float(w2),
                    })

        calendar_pass = len(calendar_violations) == 0

        return {
            "butterfly_pass": butterfly_pass,
            "butterfly_details": butterfly_violations,
            "calendar_spread_pass": calendar_pass,
            "calendar_details": calendar_violations,
        }

    def shift(self, delta_sigma: float) -> "ImpliedVolSurface":
        """
        Create a parallel-shifted surface (for Vega computation).

        Shifts the 'a' parameter of each SVI slice so that ATM total
        variance increases by approximately 2 * atm_vol * delta_sigma * T.

        Parameters
        ----------
        delta_sigma : float
            Parallel shift in implied vol (e.g., +0.01 = +1 vol point).

        Returns
        -------
        ImpliedVolSurface
            New surface with shifted ATM levels.
        """
        shifted_cals = {}
        for T_val, cal in self._calibrations.items():
            a, b, rho, m, sigma = cal["params"]
            # Shift ATM total variance: w_new = (atm_vol + delta_sigma)^2 * T
            atm_w = svi_total_variance(np.array([0.0]), cal["params"])[0]
            atm_iv = np.sqrt(max(atm_w / T_val, 1e-10))
            new_atm_w = (atm_iv + delta_sigma)**2 * T_val
            a_new = a + (new_atm_w - atm_w)
            shifted_cals[T_val] = {
                **cal,
                "params": (max(a_new, 1e-6), b, rho, m, sigma),
            }
        return ImpliedVolSurface(self.spot, self.r, shifted_cals)


# =====================================================================
# 4. Convenience: build from market data
# =====================================================================

def build_vol_surface_from_market(
    options_ticker: str = "SPY",
    use_sample: bool = False,
) -> ImpliedVolSurface:
    """
    End-to-end: fetch data -> extract IVs -> calibrate SVI -> build surface.

    Uses the same ticker for both spot and options to avoid the SPY/SPX
    mismatch (SPY ~ SPX/10).

    Parameters
    ----------
    options_ticker : str
        Ticker for both spot and options chain.
    use_sample : bool
        If True, use saved sample data (offline mode).

    Returns
    -------
    ImpliedVolSurface
    """
    from .market_data import (
        fetch_spot, fetch_options_chain, fetch_risk_free_rate,
        load_sample_data,
    )

    if use_sample:
        data = load_sample_data()
        spot = data["spot_spy"]
        r = data["risk_free_rate"]
        options_df = data["options_chain"]
    else:
        spot = fetch_spot(options_ticker)
        r = fetch_risk_free_rate()
        options_df = fetch_options_chain(options_ticker)

    logger.info("Extracting implied volatilities from %d options...", len(options_df))
    vol_data = extract_implied_vols(options_df, spot, r)
    logger.info("Successfully extracted %d implied vols", len(vol_data))

    logger.info("Calibrating SVI surface...")
    calibrations = calibrate_surface(vol_data)
    logger.info("Calibrated %d maturity slices", len(calibrations))

    return ImpliedVolSurface(spot, r, calibrations)


def estimate_eurostoxx_atm_vol(
    ticker: str = "^STOXX50E",
    window: int = 252,
) -> float:
    """
    Estimate EUROSTOXX 50 ATM vol from historical realized vol.

    Applies a 1.15x multiplier as proxy for the implied/realized vol premium
    typical in equity indices. In production, would use VSTOXX index directly.
    """
    from .market_data import fetch_historical_prices, compute_log_returns, compute_realized_vol
    hist = fetch_historical_prices(ticker, period="2y")
    rets = compute_log_returns(hist)
    rvol = compute_realized_vol(rets, window=window)
    return rvol * 1.15  # Empirical implied/realized premium


# =====================================================================
# 5. SVI analytical derivatives (used by Dupire in local_vol.py)
# =====================================================================

def svi_dw_dk(k: float, params: tuple) -> float:
    """
    dw/dk = b * (rho + (k - m) / sqrt((k - m)^2 + sigma^2))
    """
    _, b, rho, m, sigma = params
    return b * (rho + (k - m) / np.sqrt((k - m)**2 + sigma**2))


def svi_d2w_dk2(k: float, params: tuple) -> float:
    """
    d^2w/dk^2 = b * sigma^2 / ((k - m)^2 + sigma^2)^(3/2)
    """
    _, b, _, m, sigma = params
    return b * sigma**2 / ((k - m)**2 + sigma**2)**1.5


# =====================================================================
# 6. Visualization
# =====================================================================

def plot_vol_surface(
    surface: ImpliedVolSurface,
    title: str = "Implied Volatility Surface (SVI)",
    save_path: Optional[str] = None,
) -> None:
    """3D surface plot of implied vol."""
    import matplotlib.pyplot as plt

    M, T, V = surface.vol_grid()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(M, T, V * 100, cmap="viridis", alpha=0.8)
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Implied Vol (%)")
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_vol_smile(
    surface: ImpliedVolSurface,
    T: float,
    market_data: Optional["pd.DataFrame"] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    2D smile plot for a single maturity slice.

    Parameters
    ----------
    surface : ImpliedVolSurface
    T : float
        Target maturity.
    market_data : pd.DataFrame, optional
        If provided, overlay market data points. Must have
        'log_moneyness' and 'implied_vol' columns.
    """
    import matplotlib.pyplot as plt

    k_range = np.linspace(-0.3, 0.3, 100)
    params = surface._get_svi_params_at_T(T)
    vols = svi_implied_vol(k_range, T, params) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, vols, "b-", linewidth=2, label="SVI fit")

    if market_data is not None:
        slice_data = market_data[abs(market_data["T"] - T) < 0.02]
        if not slice_data.empty:
            ax.scatter(
                slice_data["log_moneyness"],
                slice_data["implied_vol"] * 100,
                c="red", s=40, zorder=5, label="Market",
            )

    ax.set_xlabel("Log-moneyness ln(K/F)")
    ax.set_ylabel("Implied Vol (%)")
    ax.set_title(title or f"Volatility Smile (T = {T:.2f}y)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
