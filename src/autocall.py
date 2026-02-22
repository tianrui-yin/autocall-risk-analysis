"""
Autocallable structured product pricing via Monte Carlo simulation.

Product: Phoenix Autocallable Note
- At each observation date:
  1. If S >= autocall_barrier -> early redemption (notional + pending memory coupons + current coupon)
  2. Else if S >= coupon_barrier -> pay coupon (+ any accumulated memory coupons)
  3. Else -> no coupon (accumulates in memory for next eligible date)
- At maturity (if not autocalled):
  - If knock-in triggered (S ever < ki_barrier) and S_T < S0: capital loss
    (coupons already paid during life are kept by investor)
  - Else: return notional (coupons already paid during life are kept)

Reference: Hull Ch. 26 (Exotic Options).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .utils import simulate_gbm_paths, simulate_gbm_paths_antithetic, simulate_paths_local_vol

if TYPE_CHECKING:
    from .local_vol import VolModel
    from .heston import HestonModel


class Autocallable:
    """
    Phoenix Autocallable Note pricer.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    autocall_barrier : float
        Autocall trigger level as fraction of S0 (e.g., 1.0 = 100%).
    coupon_barrier : float
        Coupon payment trigger as fraction of S0 (e.g., 0.8 = 80%).
    ki_barrier : float
        Knock-in put barrier as fraction of S0 (e.g., 0.6 = 60%).
    coupon_rate : float
        Coupon rate per observation period (e.g., 0.05 = 5%).
    n_observations : int
        Number of observation dates (e.g., 8 for quarterly over 2 years).
    T : float
        Maturity in years.
    r : float
        Risk-free rate (annualized).
    sigma : float
        Volatility (annualized). Used when vol_model is None.
    notional : float
        Notional amount (default 100).
    vol_model : VolModel, optional
        Volatility model (ConstantVol, TermStructureVol, LocalVol).
        If None, uses constant sigma (backwards compatible).
    """

    def __init__(
        self,
        S0: float,
        autocall_barrier: float,
        coupon_barrier: float,
        ki_barrier: float,
        coupon_rate: float,
        n_observations: int,
        T: float,
        r: float,
        sigma: float,
        notional: float = 100.0,
        vol_model: Optional[VolModel] = None,
    ):
        self._validate_params(
            S0, autocall_barrier, coupon_barrier, ki_barrier, coupon_rate, T, r, sigma
        )
        self.S0 = S0
        self.autocall_barrier = autocall_barrier * S0
        self.coupon_barrier = coupon_barrier * S0
        self.ki_barrier = ki_barrier * S0
        self.coupon_rate = coupon_rate
        self.n_observations = n_observations
        self.T = T
        self.r = r
        self.sigma = sigma
        self.notional = notional
        self.vol_model = vol_model

        self.observation_times = np.linspace(
            T / n_observations, T, n_observations
        )

    @staticmethod
    def _validate_params(S0, ac_bar, cp_bar, ki_bar, coupon, T, r, sigma):
        if S0 <= 0:
            raise ValueError("S0 must be positive")
        if not (0 < ki_bar < cp_bar <= ac_bar):
            raise ValueError(
                "Barriers must satisfy: 0 < ki_barrier < coupon_barrier <= autocall_barrier"
            )
        if coupon <= 0:
            raise ValueError("Coupon rate must be positive")
        if T <= 0:
            raise ValueError("Maturity must be positive")
        if sigma <= 0:
            raise ValueError("Volatility must be positive")

    def price(
        self,
        n_paths: int = 100_000,
        n_steps_per_period: int = 50,
        antithetic: bool = True,
        seed: int | None = None,
    ) -> dict:
        """
        Price the autocallable via Monte Carlo.

        Returns
        -------
        dict with keys:
            'price': float - present value of the product
            'std_error': float - Monte Carlo standard error
            'autocall_probs': np.ndarray - probability of autocall at each date
            'expected_life': float - expected life in years
            'ki_probability': float - probability of knock-in event
        """
        n_steps = self.n_observations * n_steps_per_period
        obs_indices = [i * n_steps_per_period for i in range(1, self.n_observations + 1)]

        # Check for HestonModel (stochastic vol — paths generated internally)
        from .heston import HestonModel
        if isinstance(self.vol_model, HestonModel):
            paths = self.vol_model.simulate_paths(
                S0=self.S0, r=self.r, T=self.T,
                n_steps=n_steps, n_paths=n_paths, seed=seed,
            )
        elif self.vol_model is not None:
            # Local vol / term structure: Euler-Maruyama simulation
            # Antithetic variates not applicable (path-dependent vol)
            paths = simulate_paths_local_vol(
                self.S0, self.r, self.vol_model, self.T, n_steps, n_paths, seed
            )
        elif antithetic:
            paths = simulate_gbm_paths_antithetic(
                self.S0, self.r, self.sigma, self.T, n_steps, n_paths, seed
            )
        else:
            paths = simulate_gbm_paths(
                self.S0, self.r, self.sigma, self.T, n_steps, n_paths, seed
            )

        payoffs = np.zeros(n_paths)
        autocalled = np.zeros(n_paths, dtype=bool)
        autocall_dates = np.full(n_paths, np.nan)
        ki_triggered = np.zeros(n_paths, dtype=bool)
        autocall_counts = np.zeros(self.n_observations)

        path_min = np.min(paths, axis=0)
        ki_triggered = path_min < self.ki_barrier

        # Memory coupon tracking: number of consecutive unpaid coupon periods
        pending_coupons = np.zeros(n_paths, dtype=int)

        for i, obs_idx in enumerate(obs_indices):
            obs_time = self.observation_times[i]
            S_obs = paths[obs_idx]
            discount = np.exp(-self.r * obs_time)
            alive = ~autocalled

            # --- Autocall check ---
            can_autocall = alive & (S_obs >= self.autocall_barrier)
            if np.any(can_autocall):
                # Pay notional + accumulated memory coupons + current coupon
                n_cpn = pending_coupons[can_autocall] + 1
                payoffs[can_autocall] += self.notional * (1 + self.coupon_rate * n_cpn) * discount
                autocall_dates[can_autocall] = obs_time
                autocalled[can_autocall] = True
                autocall_counts[i] = np.sum(can_autocall)

            # --- Coupon check (memory feature) for non-autocalled alive paths ---
            still_alive = alive & ~can_autocall
            cpn_paid = still_alive & (S_obs >= self.coupon_barrier)
            cpn_missed = still_alive & (S_obs < self.coupon_barrier)

            if np.any(cpn_paid):
                n_cpn = pending_coupons[cpn_paid] + 1
                payoffs[cpn_paid] += self.notional * self.coupon_rate * n_cpn * discount
                pending_coupons[cpn_paid] = 0

            pending_coupons[cpn_missed] += 1

        # --- Terminal payoff for non-autocalled paths ---
        still_alive = ~autocalled
        S_T = paths[-1]
        discount_T = np.exp(-self.r * self.T)

        ki_loss = still_alive & ki_triggered & (S_T < self.S0)
        no_ki = still_alive & ~ki_loss

        # KI loss: capital return only (coupons already paid in loop)
        payoffs[ki_loss] += self.notional * (S_T[ki_loss] / self.S0) * discount_T

        # No KI: full notional returned (coupons already paid in loop)
        payoffs[no_ki] += self.notional * discount_T

        price_mean = np.mean(payoffs)
        price_std = np.std(payoffs) / np.sqrt(n_paths)

        autocall_probs = autocall_counts / n_paths

        expected_life_arr = np.where(
            autocalled, autocall_dates, self.T
        )
        expected_life = np.mean(expected_life_arr)

        return {
            "price": price_mean,
            "std_error": price_std,
            "autocall_probs": autocall_probs,
            "total_autocall_prob": np.sum(autocall_probs),
            "expected_life": expected_life,
            "ki_probability": np.mean(ki_triggered),
            "ki_loss_probability": np.mean(ki_loss),
        }

    def price_for_greeks(
        self,
        S0: float | None = None,
        sigma: float | None = None,
        r: float | None = None,
        T: float | None = None,
        n_paths: int = 100_000,
        seed: int | None = None,
    ) -> float:
        """
        Reprice with bumped parameters. Used by Greeks finite difference.
        Barriers remain fixed (set at inception), only spot/vol/rate/T change.
        Returns scalar price only.

        When vol_model is set, bumping sigma creates a parallel-shifted
        vol model (all implied vols shift by the same amount).
        """
        saved = {
            "S0": self.S0, "sigma": self.sigma,
            "r": self.r, "T": self.T,
            "obs_times": self.observation_times.copy(),
            "vol_model": self.vol_model,
        }

        if S0 is not None:
            self.S0 = S0
        if sigma is not None:
            if self.vol_model is not None and hasattr(self.vol_model, "shift"):
                # Parallel shift the vol model
                delta = sigma - self.sigma
                self.vol_model = self.vol_model.shift(delta)
            self.sigma = sigma
        if r is not None:
            self.r = r
        if T is not None:
            self.T = T
            self.observation_times = np.linspace(
                T / self.n_observations, T, self.n_observations
            )

        result = self.price(n_paths=n_paths, seed=seed)
        price = result["price"]

        self.S0 = saved["S0"]
        self.sigma = saved["sigma"]
        self.r = saved["r"]
        self.T = saved["T"]
        self.observation_times = saved["obs_times"]
        self.vol_model = saved["vol_model"]

        return price

    def description(self) -> str:
        vol_info = self.vol_model.model_name if self.vol_model else f"σ={self.sigma:.1%}"
        return (
            f"Phoenix Autocallable | S0={self.S0:.0f} | "
            f"AC={self.autocall_barrier:.0f} ({self.autocall_barrier/self.S0:.0%}) | "
            f"KI={self.ki_barrier:.0f} ({self.ki_barrier/self.S0:.0%}) | "
            f"Coupon={self.coupon_rate:.1%}/period | "
            f"T={self.T:.1f}y | {vol_info}"
        )
