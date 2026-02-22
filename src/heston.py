"""
Heston stochastic volatility model for pricing and calibration.

Heston (1993) SDE:
    dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW_1
    dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_2
    corr(dW_1, dW_2) = rho

The model captures:
- Stochastic volatility (smile dynamics)
- Leverage effect via rho < 0 (spot-vol correlation)
- Mean-reverting variance (kappa, theta)
- Fat tails via vol-of-vol (xi)

Semi-analytical pricing via characteristic function (Lewis 2000).
Calibration via differential evolution (global optimizer).

Reference:
- Heston, S. "A Closed-Form Solution for Options with Stochastic Volatility"
  (Review of Financial Studies, 1993).
- Gatheral, J. "The Volatility Surface" (Wiley, 2006), Ch. 2 & 4.
- Lewis, A. "Option Valuation under Stochastic Volatility" (Finance Press, 2000).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import integrate, optimize

logger = logging.getLogger(__name__)


class HestonModel:
    """
    Heston stochastic volatility model.

    Parameters
    ----------
    kappa : float
        Mean reversion speed of variance.
    theta : float
        Long-term variance level.
    xi : float
        Volatility of variance (vol of vol).
    v0 : float
        Initial variance.
    rho : float
        Correlation between spot and variance Brownian motions.
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        xi: float,
        v0: float,
        rho: float,
    ):
        self._validate_params(kappa, theta, xi, v0, rho)
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.v0 = v0
        self.rho = rho

        if not self.feller_satisfied:
            logger.warning(
                "Feller condition violated: 2*kappa*theta=%.4f < xi^2=%.4f. "
                "Variance process may reach zero.",
                2 * kappa * theta, xi**2,
            )

    @staticmethod
    def _validate_params(kappa, theta, xi, v0, rho):
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if theta <= 0:
            raise ValueError("theta must be positive")
        if xi <= 0:
            raise ValueError("xi must be positive")
        if v0 <= 0:
            raise ValueError("v0 must be positive")
        if not (-1.0 < rho < 1.0):
            raise ValueError("rho must be in (-1, 1)")

    @property
    def feller_satisfied(self) -> bool:
        """Check 2*kappa*theta > xi^2 (variance stays strictly positive)."""
        return 2 * self.kappa * self.theta > self.xi**2

    @property
    def model_name(self) -> str:
        return (
            f"Heston(kappa={self.kappa:.2f}, theta={self.theta:.4f}, "
            f"xi={self.xi:.2f}, v0={self.v0:.4f}, rho={self.rho:.2f})"
        )

    # =================================================================
    # 1. Monte Carlo Path Simulation
    # =================================================================

    def simulate_paths(
        self,
        S0: float,
        r: float,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: int | None = None,
        return_variance: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston paths via Euler-Maruyama on log(S) and v.

        Uses absorption scheme for negative variance: v_t = max(v_t, 0).
        Correlated Brownians via Cholesky: Z2 = rho*Z1 + sqrt(1-rho^2)*W2.

        Parameters
        ----------
        S0 : float
            Initial spot price.
        r : float
            Risk-free rate.
        T : float
            Time horizon in years.
        n_steps : int
            Number of time steps.
        n_paths : int
            Number of Monte Carlo paths.
        seed : int, optional
            Random seed for reproducibility.
        return_variance : bool
            If True, also return variance paths.

        Returns
        -------
        np.ndarray or tuple
            Price paths of shape (n_steps+1, n_paths).
            If return_variance=True, also returns variance paths.
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps

        log_S = np.zeros((n_steps + 1, n_paths))
        v = np.zeros((n_steps + 1, n_paths))

        log_S[0] = np.log(S0)
        v[0] = self.v0

        sqrt_one_minus_rho2 = np.sqrt(1.0 - self.rho**2)

        for step in range(n_steps):
            Z1 = rng.standard_normal(n_paths)
            W2 = rng.standard_normal(n_paths)
            Z2 = self.rho * Z1 + sqrt_one_minus_rho2 * W2

            v_current = v[step]
            # Absorption: clamp negative variance to zero
            v_pos = np.maximum(v_current, 0.0)
            sqrt_v = np.sqrt(v_pos)

            # Euler-Maruyama on log(S):
            # d(log S) = (r - 0.5*v) dt + sqrt(v) dW1
            log_S[step + 1] = (
                log_S[step]
                + (r - 0.5 * v_pos) * dt
                + sqrt_v * np.sqrt(dt) * Z1
            )

            # Euler-Maruyama on v:
            # dv = kappa*(theta - v) dt + xi*sqrt(v) dW2
            v[step + 1] = (
                v_current
                + self.kappa * (self.theta - v_pos) * dt
                + self.xi * sqrt_v * np.sqrt(dt) * Z2
            )
            # Absorption scheme: clamp to zero
            v[step + 1] = np.maximum(v[step + 1], 0.0)

        paths = np.exp(log_S)

        if return_variance:
            return paths, v
        return paths

    # =================================================================
    # 2. Semi-Analytical Pricing via Characteristic Function
    # =================================================================

    def _characteristic_function(
        self, u: complex, T: float, r: float,
    ) -> complex:
        """
        Heston risk-neutral characteristic function:
            phi(u) = E^Q[ exp(i*u*x_T) ]
        where x_T = ln(S_T).

        Uses the "little Heston trap" fix (Albrecher et al., 2007):
        always pick d with Re(d) > 0 and use g = (b-d)/(b+d) with |g|<1,
        then work with exp(-dT) which decays.
        """
        kappa, theta, xi, v0, rho = (
            self.kappa, self.theta, self.xi, self.v0, self.rho,
        )
        xi2 = xi * xi
        iu = 1j * u

        # Standard Heston cf (log price, risk-neutral)
        # b = kappa - rho*xi*iu
        b = kappa - rho * xi * iu
        # d = sqrt(b^2 + xi^2*(iu + u^2))
        d = np.sqrt(b * b + xi2 * (iu + u * u))

        # Ensure Re(d) > 0
        if d.real < 0:
            d = -d

        g = (b - d) / (b + d)
        exp_neg_dT = np.exp(-d * T)

        D = (b - d) / xi2 * (1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT)

        C = iu * r * T + (kappa * theta / xi2) * (
            (b - d) * T - 2.0 * np.log((1.0 - g * exp_neg_dT) / (1.0 - g))
        )

        return np.exp(C + D * v0 + iu * np.log(self._S0_cache))

    def call_price(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
    ) -> float:
        """
        European call price via Carr-Madan / Gil-Pelaez inversion.

        Uses the standard decomposition:
            C = S0*P1 - K*e^{-rT}*P2

        where P1, P2 are computed from the risk-neutral characteristic function
        via Gil-Pelaez:
            Pj = 0.5 + (1/pi) * integral_0^inf Re[...] du

        Implementation uses a single characteristic function with the
        Albrecher et al. (2007) formulation for numerical stability.

        Parameters
        ----------
        S0, K, T, r : float
            Spot, strike, maturity, risk-free rate.

        Returns
        -------
        float
            European call price.
        """
        # Cache S0 for the characteristic function
        self._S0_cache = S0
        log_K = np.log(K)

        def integrand_P1(u: float) -> float:
            """Integrand for P1 (stock-price measure)."""
            if u < 1e-12:
                return 0.0
            phi = self._characteristic_function(u - 1j, T, r)
            # P1: Re[ e^{-iu*lnK} * phi(u-i) / (iu * phi(-i)) ]
            # phi(-i) = E[S_T/S_0 * e^{rT}] ... but simpler:
            # P1 = 0.5 + 1/pi * Re[ integral e^{-iu*lnK} * phi(u-i) / (iu * S0 * e^{rT}) du ]
            # Actually use: P1 from Gil-Pelaez
            val = np.real(
                np.exp(-1j * u * log_K) * phi / (1j * u * S0 * np.exp(r * T))
            )
            return float(val)

        def integrand_P2(u: float) -> float:
            """Integrand for P2 (risk-neutral measure)."""
            if u < 1e-12:
                return 0.0
            phi = self._characteristic_function(u, T, r)
            val = np.real(np.exp(-1j * u * log_K) * phi / (1j * u))
            return float(val)

        I1, _ = integrate.quad(integrand_P1, 1e-8, 200, limit=200)
        I2, _ = integrate.quad(integrand_P2, 1e-8, 200, limit=200)

        P1 = 0.5 + I1 / np.pi
        P2 = 0.5 + I2 / np.pi

        price = S0 * P1 - K * np.exp(-r * T) * P2
        return max(float(price), 0.0)

    def put_price(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
    ) -> float:
        """European put price via put-call parity."""
        call = self.call_price(S0, K, T, r)
        return call - S0 + K * np.exp(-r * T)

    # =================================================================
    # 3. Calibration
    # =================================================================

    @classmethod
    def calibrate(
        cls,
        market_prices: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        spot: float,
        r: float,
        bounds: Optional[dict] = None,
        maxiter: int = 200,
        seed: int = 42,
    ) -> dict:
        """
        Calibrate Heston parameters to market option prices.

        Uses scipy.optimize.differential_evolution for global search.

        Parameters
        ----------
        market_prices : np.ndarray
            Observed call prices.
        strikes : np.ndarray
            Strike prices.
        maturities : np.ndarray
            Maturities (must match market_prices elementwise).
        spot : float
            Current spot price.
        r : float
            Risk-free rate.
        bounds : dict, optional
            Parameter bounds. Defaults to typical equity ranges.
        maxiter : int
            Maximum iterations for optimizer.
        seed : int
            Random seed for differential evolution.

        Returns
        -------
        dict with keys:
            'model': HestonModel — calibrated model
            'success': bool
            'rmse': float — root mean squared pricing error
            'max_error': float — maximum absolute pricing error
            'params': dict — calibrated parameter values
        """
        market_prices = np.asarray(market_prices, dtype=float)
        strikes = np.asarray(strikes, dtype=float)
        maturities = np.asarray(maturities, dtype=float)

        if bounds is None:
            param_bounds = [
                (0.1, 10.0),     # kappa
                (0.005, 0.2),    # theta
                (0.05, 1.5),     # xi
                (0.005, 0.2),    # v0
                (-0.95, -0.05),  # rho (typically negative for equity)
            ]
        else:
            param_bounds = [
                bounds.get("kappa", (0.1, 10.0)),
                bounds.get("theta", (0.005, 0.2)),
                bounds.get("xi", (0.05, 1.5)),
                bounds.get("v0", (0.005, 0.2)),
                bounds.get("rho", (-0.95, -0.05)),
            ]

        def objective(params):
            kappa, theta, xi, v0, rho = params
            try:
                model = cls(kappa=kappa, theta=theta, xi=xi, v0=v0, rho=rho)
            except ValueError:
                return 1e10

            errors = np.zeros(len(market_prices))
            for i in range(len(market_prices)):
                try:
                    model_price = model.call_price(
                        S0=spot, K=strikes[i], T=maturities[i], r=r,
                    )
                    errors[i] = (model_price - market_prices[i]) ** 2
                except Exception:
                    errors[i] = 1e6
            return float(np.sum(errors))

        result = optimize.differential_evolution(
            objective,
            bounds=param_bounds,
            maxiter=maxiter,
            seed=seed,
            tol=1e-8,
            polish=True,
        )

        kappa, theta, xi, v0, rho = result.x
        calibrated_model = cls(kappa=kappa, theta=theta, xi=xi, v0=v0, rho=rho)

        # Compute fit metrics
        model_prices = np.array([
            calibrated_model.call_price(S0=spot, K=strikes[i], T=maturities[i], r=r)
            for i in range(len(market_prices))
        ])
        abs_errors = np.abs(model_prices - market_prices)
        rmse = float(np.sqrt(np.mean(abs_errors**2)))
        max_error = float(np.max(abs_errors))

        logger.info(
            "Heston calibration: kappa=%.3f, theta=%.5f, xi=%.3f, v0=%.5f, rho=%.3f | "
            "RMSE=%.4f, MaxErr=%.4f",
            kappa, theta, xi, v0, rho, rmse, max_error,
        )

        return {
            "model": calibrated_model,
            "success": result.success,
            "rmse": rmse,
            "max_error": max_error,
            "params": {
                "kappa": kappa, "theta": theta, "xi": xi,
                "v0": v0, "rho": rho,
            },
        }
