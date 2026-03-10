"""
SABR stochastic volatility model calibration (Hagan et al., 2002).

The SABR model is a stochastic volatility model with the SDEs:
    dF = alpha * F^beta * dW_1
    dalpha = nu * alpha * dW_2
    corr(dW_1, dW_2) = rho

Rather than simulating paths, we use Hagan's closed-form approximation
for the implied volatility, which is standard practice for calibration
and smile interpolation.

For beta=1 (lognormal SABR, standard for equity):
    sigma_imp(K, F, T) = alpha * (z / x(z)) * [1 + correction * T]

where:
    z = (nu / alpha) * ln(F / K)
    x(z) = ln((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho))
    correction = rho*nu*alpha/4 + (2 - 3*rho^2)*nu^2/24

When F = K (ATM): z = 0, z/x(z) = 1, so sigma_ATM = alpha * [1 + correction * T]

Reference:
- Hagan, P. et al. "Managing Smile Risk" (Wilmott Magazine, 2002).
- Gatheral, J. "The Volatility Surface" (Wiley, 2006), Ch. 4.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)


class SABRModel:
    """
    SABR stochastic volatility model (Hagan et al., 2002).

    Uses the Hagan implied vol approximation (no MC needed).
    beta is fixed at 1.0 (lognormal SABR, standard for equity markets).

    Parameters
    ----------
    alpha : float
        Initial vol level (ATM vol ~ alpha for short T).
    rho : float
        Correlation between spot and vol (-1 < rho < 1).
    nu : float
        Vol of vol (volvol).
    beta : float
        CEV exponent (fixed at 1.0).
    """

    def __init__(self, alpha: float, rho: float, nu: float, beta: float = 1.0):
        self._validate_params(alpha, rho, nu, beta)
        self.alpha = alpha
        self.rho = rho
        self.nu = nu
        self.beta = beta

    @staticmethod
    def _validate_params(alpha: float, rho: float, nu: float, beta: float) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not (-1.0 < rho < 1.0):
            raise ValueError("rho must be in (-1, 1)")
        if nu <= 0:
            raise ValueError("nu must be positive")
        if beta < 0 or beta > 1:
            raise ValueError("beta must be in [0, 1]")

    @property
    def model_name(self) -> str:
        return (
            f"SABR(alpha={self.alpha:.2f}, rho={self.rho:.2f}, "
            f"nu={self.nu:.2f}, beta={self.beta:.1f})"
        )

    def implied_vol(self, F: float, K: float, T: float) -> float:
        """
        Hagan approximation for SABR implied vol (beta=1 lognormal case).

        For beta=1, the general Hagan formula simplifies to:

            sigma_imp = alpha * (z / x(z)) * [1 + C * T]

        where:
            z = (nu / alpha) * ln(F / K)
            x(z) = ln((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho))
            C = rho*nu*alpha/4 + (2 - 3*rho^2)*nu^2/24

        When K = F (ATM), z = 0 and z/x(z) -> 1 by L'Hopital.

        Parameters
        ----------
        F : float
            Forward price.
        K : float
            Strike price.
        T : float
            Time to maturity.

        Returns
        -------
        float
            SABR implied volatility.
        """
        alpha = self.alpha
        rho = self.rho
        nu = self.nu

        # Correction term (same for ATM and non-ATM)
        correction = 1.0 + (rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho**2) * nu**2 / 24.0) * T

        # ATM case: K ≈ F
        log_FK = np.log(F / K)
        if abs(log_FK) < 1e-12:
            return float(max(alpha * correction, 1e-10))

        # Non-ATM case
        z = (nu / alpha) * log_FK

        # Compute x(z) = ln((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho))
        discriminant = 1.0 - 2.0 * rho * z + z**2
        # Guard against numerical negativity
        discriminant = max(discriminant, 1e-16)
        sqrt_disc = np.sqrt(discriminant)

        numerator = sqrt_disc + z - rho
        denominator = 1.0 - rho

        # Guard against log of non-positive
        if numerator / denominator <= 0:
            # Fallback: use ATM approximation
            return float(max(alpha * correction, 1e-10))

        xz = np.log(numerator / denominator)

        # Guard against x(z) = 0
        if abs(xz) < 1e-16:
            return float(max(alpha * correction, 1e-10))

        # z / x(z) ratio
        z_over_xz = z / xz

        vol = alpha * z_over_xz * correction
        return float(max(vol, 1e-10))

    @classmethod
    def calibrate(
        cls,
        market_vols: np.ndarray,
        strikes: np.ndarray,
        F: float,
        T: float,
        beta: float = 1.0,
    ) -> dict:
        """
        Calibrate alpha, rho, nu to market implied vols for a single maturity slice.

        Uses scipy.optimize.minimize (L-BFGS-B) minimizing sum of squared vol errors,
        with differential_evolution fallback.

        Parameters
        ----------
        market_vols : np.ndarray
            Market implied volatilities.
        strikes : np.ndarray
            Strike prices.
        F : float
            Forward price.
        T : float
            Time to maturity.
        beta : float
            CEV exponent (default 1.0).

        Returns
        -------
        dict with keys:
            'model': SABRModel  -- calibrated model
            'rmse': float       -- root mean squared vol error
            'params': dict      -- calibrated parameter values
        """
        market_vols = np.asarray(market_vols, dtype=float)
        strikes = np.asarray(strikes, dtype=float)

        # Initial guess: alpha ~ ATM vol, rho ~ -0.3 (typical equity), nu ~ 0.3
        atm_idx = np.argmin(np.abs(strikes - F))
        alpha0 = float(market_vols[atm_idx])
        x0 = [alpha0, -0.3, 0.3]

        bounds = [
            (0.001, 2.0),      # alpha
            (-0.999, 0.999),   # rho
            (0.001, 3.0),      # nu
        ]

        def objective(params):
            alpha, rho, nu = params
            try:
                model = cls(alpha=alpha, rho=rho, nu=nu, beta=beta)
                model_vols = np.array([model.implied_vol(F, K, T) for K in strikes])
                return float(np.sum((model_vols - market_vols) ** 2))
            except (ValueError, RuntimeWarning):
                return 1e10

        # L-BFGS-B
        best_result = None
        try:
            result = optimize.minimize(
                objective, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-14},
            )
            if result.success or result.fun < 1e-4:
                best_result = result
        except Exception as exc:
            logger.debug("L-BFGS-B failed: %s", exc)

        # Differential evolution fallback
        if best_result is None or best_result.fun > 1e-6:
            try:
                de_result = optimize.differential_evolution(
                    objective, bounds, maxiter=300, seed=42, tol=1e-12,
                )
                if best_result is None or de_result.fun < best_result.fun:
                    best_result = de_result
            except Exception as exc:
                logger.warning("Differential evolution failed: %s", exc)

        if best_result is None:
            # Last resort: return initial guess
            logger.warning("SABR calibration failed, using initial guess")
            best_result = type("R", (), {"x": x0})()

        alpha_cal, rho_cal, nu_cal = best_result.x
        calibrated_model = cls(alpha=alpha_cal, rho=rho_cal, nu=nu_cal, beta=beta)

        # Compute RMSE
        model_vols = np.array([calibrated_model.implied_vol(F, K, T) for K in strikes])
        rmse = float(np.sqrt(np.mean((model_vols - market_vols) ** 2)))

        logger.info(
            "SABR calibration T=%.3f: alpha=%.4f, rho=%.3f, nu=%.3f | RMSE=%.6f",
            T, alpha_cal, rho_cal, nu_cal, rmse,
        )

        return {
            "model": calibrated_model,
            "rmse": rmse,
            "params": {
                "alpha": alpha_cal,
                "rho": rho_cal,
                "nu": nu_cal,
            },
        }

    @classmethod
    def calibrate_surface(
        cls,
        vol_data: "pd.DataFrame",
        spot: float,
        r: float,
        beta: float = 1.0,
    ) -> dict:
        """
        Calibrate SABR per maturity slice (analogous to SVI calibrate_surface).

        Parameters
        ----------
        vol_data : pd.DataFrame
            Must have columns 'T', 'K' (or 'strike'), 'implied_vol'.
        spot : float
            Current spot price.
        r : float
            Risk-free rate.
        beta : float
            CEV exponent (default 1.0).

        Returns
        -------
        dict mapping T -> calibration result dict.
        """
        import pandas as pd

        # Determine strike column name
        if "K" in vol_data.columns:
            strike_col = "K"
        elif "strike" in vol_data.columns:
            strike_col = "strike"
        else:
            raise ValueError("vol_data must have a 'K' or 'strike' column")

        calibrations = {}

        for T_val, group in vol_data.groupby("T"):
            if len(group) < 3:
                logger.info(
                    "Skipping T=%.3f: only %d points (need >= 3)", T_val, len(group)
                )
                continue

            F = spot * np.exp(r * T_val)
            strikes = group[strike_col].values
            impl_vols = group["implied_vol"].values

            result = cls.calibrate(
                market_vols=impl_vols,
                strikes=strikes,
                F=F,
                T=T_val,
                beta=beta,
            )

            calibrations[T_val] = result
            logger.info(
                "T=%.3f: SABR calibrated (RMSE=%.6f)", T_val, result["rmse"]
            )

        return dict(sorted(calibrations.items()))
