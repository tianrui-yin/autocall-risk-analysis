"""Tests for Heston stochastic volatility model."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.heston import HestonModel


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def default_heston():
    """Typical equity Heston parameters."""
    return HestonModel(
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        v0=0.04,
        rho=-0.7,
    )


@pytest.fixture
def low_vol_of_vol_heston():
    """Heston with xi -> 0 (approaches Black-Scholes)."""
    return HestonModel(
        kappa=2.0,
        theta=0.04,
        xi=1e-6,
        v0=0.04,
        rho=-0.7,
    )


# =====================================================================
# Parameter Validation
# =====================================================================

class TestHestonValidation:
    def test_negative_kappa_raises(self):
        with pytest.raises(ValueError, match="kappa"):
            HestonModel(kappa=-1.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7)

    def test_negative_theta_raises(self):
        with pytest.raises(ValueError, match="theta"):
            HestonModel(kappa=2.0, theta=-0.01, xi=0.3, v0=0.04, rho=-0.7)

    def test_negative_xi_raises(self):
        with pytest.raises(ValueError, match="xi"):
            HestonModel(kappa=2.0, theta=0.04, xi=-0.1, v0=0.04, rho=-0.7)

    def test_negative_v0_raises(self):
        with pytest.raises(ValueError, match="v0"):
            HestonModel(kappa=2.0, theta=0.04, xi=0.3, v0=-0.01, rho=-0.7)

    def test_rho_out_of_range_raises(self):
        with pytest.raises(ValueError, match="rho"):
            HestonModel(kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=1.5)
        with pytest.raises(ValueError, match="rho"):
            HestonModel(kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-1.5)


class TestFellerCondition:
    def test_feller_satisfied(self):
        """2*kappa*theta > xi^2 should be True."""
        model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7)
        # 2*2.0*0.04 = 0.16 > 0.09 = 0.3^2
        assert model.feller_satisfied

    def test_feller_violated(self):
        """High vol-of-vol violates Feller."""
        model = HestonModel(kappa=0.5, theta=0.02, xi=0.8, v0=0.04, rho=-0.7)
        # 2*0.5*0.02 = 0.02 < 0.64 = 0.8^2
        assert not model.feller_satisfied

    def test_feller_warning(self, caplog):
        """Should log warning when Feller violated."""
        import logging
        with caplog.at_level(logging.WARNING):
            HestonModel(kappa=0.5, theta=0.02, xi=0.8, v0=0.04, rho=-0.7)
        assert "Feller" in caplog.text


# =====================================================================
# Path Simulation
# =====================================================================

class TestHestonPaths:
    def test_path_shape(self, default_heston):
        paths = default_heston.simulate_paths(
            S0=100, r=0.03, T=1.0, n_steps=100, n_paths=500, seed=42,
        )
        assert paths.shape == (101, 500)

    def test_paths_start_at_S0(self, default_heston):
        paths = default_heston.simulate_paths(
            S0=100, r=0.03, T=1.0, n_steps=100, n_paths=500, seed=42,
        )
        assert np.allclose(paths[0], 100.0)

    def test_paths_positive(self, default_heston):
        """All simulated prices must be positive (log-space simulation)."""
        paths = default_heston.simulate_paths(
            S0=100, r=0.03, T=1.0, n_steps=252, n_paths=5000, seed=42,
        )
        assert np.all(paths > 0)

    def test_paths_reasonable_range(self, default_heston):
        """Terminal prices should be in a reasonable range for 1y horizon."""
        paths = default_heston.simulate_paths(
            S0=100, r=0.03, T=1.0, n_steps=252, n_paths=10000, seed=42,
        )
        S_T = paths[-1]
        assert np.mean(S_T) > 50
        assert np.mean(S_T) < 200
        assert np.std(S_T) > 5  # not degenerate

    def test_reproducibility(self, default_heston):
        """Same seed -> same paths."""
        p1 = default_heston.simulate_paths(S0=100, r=0.03, T=1.0, n_steps=50, n_paths=100, seed=123)
        p2 = default_heston.simulate_paths(S0=100, r=0.03, T=1.0, n_steps=50, n_paths=100, seed=123)
        assert np.allclose(p1, p2)

    def test_different_seeds_differ(self, default_heston):
        p1 = default_heston.simulate_paths(S0=100, r=0.03, T=1.0, n_steps=50, n_paths=100, seed=1)
        p2 = default_heston.simulate_paths(S0=100, r=0.03, T=1.0, n_steps=50, n_paths=100, seed=2)
        assert not np.allclose(p1, p2)

    def test_variance_non_negative(self, default_heston):
        """Variance paths should remain non-negative (absorption scheme)."""
        paths, var_paths = default_heston.simulate_paths(
            S0=100, r=0.03, T=1.0, n_steps=252, n_paths=5000, seed=42,
            return_variance=True,
        )
        assert np.all(var_paths >= 0)

    def test_convergence_to_risk_neutral_drift(self, default_heston):
        """E[S_T] should be approximately S0 * exp(r*T) under risk-neutral measure."""
        S0, r, T = 100, 0.05, 1.0
        paths = default_heston.simulate_paths(
            S0=S0, r=r, T=T, n_steps=252, n_paths=100_000, seed=42,
        )
        expected_mean = S0 * np.exp(r * T)
        actual_mean = np.mean(paths[-1])
        # Allow 1% relative error due to MC
        assert abs(actual_mean - expected_mean) / expected_mean < 0.01


# =====================================================================
# Heston Call Pricing (Semi-Analytical)
# =====================================================================

class TestHestonCallPrice:
    def test_call_price_positive(self, default_heston):
        """ATM call should have positive price."""
        price = default_heston.call_price(S0=100, K=100, T=1.0, r=0.03)
        assert price > 0

    def test_call_price_bounded(self, default_heston):
        """Call price <= S0 and >= max(S - K*exp(-rT), 0)."""
        S0, K, T, r = 100, 100, 1.0, 0.03
        price = default_heston.call_price(S0=S0, K=K, T=T, r=r)
        intrinsic = max(S0 - K * np.exp(-r * T), 0)
        assert price >= intrinsic - 0.01
        assert price <= S0 + 0.01

    def test_heston_reduces_to_bs_when_xi_zero(self, low_vol_of_vol_heston):
        """When xi -> 0, Heston price should converge to Black-Scholes."""
        from src.vol_surface import bs_call_price
        S0, K, T, r = 100, 100, 1.0, 0.03
        sigma = np.sqrt(low_vol_of_vol_heston.v0)  # sqrt(0.04) = 0.20

        heston_price = low_vol_of_vol_heston.call_price(S0=S0, K=K, T=T, r=r)
        bs_price = bs_call_price(S0, K, T, r, sigma)

        assert abs(heston_price - bs_price) < 0.5  # within 0.50

    def test_higher_vol_of_vol_increases_otm_put(self):
        """Higher xi with rho<0 increases OTM put prices (downside skew)."""
        low_xi = HestonModel(kappa=2.0, theta=0.04, xi=0.1, v0=0.04, rho=-0.7)
        high_xi = HestonModel(kappa=2.0, theta=0.04, xi=0.5, v0=0.04, rho=-0.7)

        # OTM put: K < S0 (left wing of the smile, amplified by rho<0)
        p_low = low_xi.put_price(S0=100, K=80, T=1.0, r=0.03)
        p_high = high_xi.put_price(S0=100, K=80, T=1.0, r=0.03)
        assert p_high > p_low

    def test_put_call_parity(self, default_heston):
        """C - P = S - K*exp(-rT)."""
        S0, K, T, r = 100, 100, 1.0, 0.03
        call = default_heston.call_price(S0=S0, K=K, T=T, r=r)
        put = default_heston.put_price(S0=S0, K=K, T=T, r=r)
        parity_rhs = S0 - K * np.exp(-r * T)
        assert abs((call - put) - parity_rhs) < 0.1


# =====================================================================
# Calibration
# =====================================================================

class TestHestonCalibration:
    def test_calibrate_on_synthetic_data(self):
        """Calibrate Heston to its own prices and recover parameters."""
        true_model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7)
        S0, r = 100, 0.03

        # Generate synthetic market prices
        strikes = np.array([80, 90, 95, 100, 105, 110, 120])
        maturities = np.array([0.25, 0.5, 1.0])

        market_prices = []
        strike_list = []
        maturity_list = []
        for T in maturities:
            for K in strikes:
                price = true_model.call_price(S0=S0, K=K, T=T, r=r)
                market_prices.append(price)
                strike_list.append(K)
                maturity_list.append(T)

        result = HestonModel.calibrate(
            market_prices=np.array(market_prices),
            strikes=np.array(strike_list),
            maturities=np.array(maturity_list),
            spot=S0,
            r=r,
        )

        # RMSE is the reliable metric (success flag may be False even with
        # perfect fit if differential_evolution hits max iterations)
        assert result["rmse"] < 0.5  # RMSE in price space < 0.50
        cal_model = result["model"]
        # Calibrated params should be in reasonable range
        assert 0.5 <= cal_model.kappa <= 10.0
        assert 0.01 <= cal_model.theta <= 0.2
        assert abs(cal_model.rho - (-0.7)) < 0.3  # rho roughly recovered

    def test_calibration_returns_fit_metrics(self):
        """Calibration should return RMSE and max error."""
        model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7)
        S0, r = 100, 0.03
        strikes = np.array([90, 100, 110])
        maturities = np.array([0.5, 1.0])

        prices = []
        K_list, T_list = [], []
        for T in maturities:
            for K in strikes:
                prices.append(model.call_price(S0=S0, K=K, T=T, r=r))
                K_list.append(K)
                T_list.append(T)

        result = HestonModel.calibrate(
            market_prices=np.array(prices),
            strikes=np.array(K_list),
            maturities=np.array(T_list),
            spot=S0,
            r=r,
        )
        assert "rmse" in result
        assert "max_error" in result
        assert result["rmse"] >= 0
        assert result["max_error"] >= 0


# =====================================================================
# Integration with Autocallable
# =====================================================================

class TestHestonAutocallIntegration:
    def test_autocall_with_heston_runs(self, default_heston):
        """Autocallable pricing with HestonModel should run without error."""
        from src.autocall import Autocallable
        ac = Autocallable(
            S0=100, autocall_barrier=1.0, coupon_barrier=0.8,
            ki_barrier=0.6, coupon_rate=0.05, n_observations=4,
            T=1.0, r=0.03, sigma=0.20,
            vol_model=default_heston,
        )
        result = ac.price(n_paths=10_000, seed=42)
        assert result["price"] > 0
        assert 0 <= result["total_autocall_prob"] <= 1.0
        assert 0 <= result["ki_probability"] <= 1.0

    def test_heston_price_differs_from_constant(self, default_heston):
        """Heston and constant vol should give different autocall prices."""
        from src.autocall import Autocallable

        kwargs = dict(
            S0=100, autocall_barrier=1.0, coupon_barrier=0.8,
            ki_barrier=0.6, coupon_rate=0.05, n_observations=4,
            T=1.0, r=0.03, sigma=0.20,
        )

        ac_const = Autocallable(**kwargs)
        ac_heston = Autocallable(**kwargs, vol_model=default_heston)

        r_const = ac_const.price(n_paths=50_000, seed=42)
        r_heston = ac_heston.price(n_paths=50_000, seed=42)

        # Prices should differ (Heston has vol smile / stochastic vol)
        assert abs(r_const["price"] - r_heston["price"]) > 0.01

    def test_heston_autocall_reasonable_price(self, default_heston):
        """Heston autocall price should be in reasonable range."""
        from src.autocall import Autocallable
        ac = Autocallable(
            S0=100, autocall_barrier=1.0, coupon_barrier=0.8,
            ki_barrier=0.6, coupon_rate=0.05, n_observations=8,
            T=2.0, r=0.03, sigma=0.20,
            vol_model=default_heston,
        )
        result = ac.price(n_paths=50_000, seed=42)
        # Price should be between 70 and 140 (notional=100)
        assert 70 < result["price"] < 140

    def test_heston_convergence_with_paths(self, default_heston):
        """More paths should reduce std error."""
        from src.autocall import Autocallable
        ac = Autocallable(
            S0=100, autocall_barrier=1.0, coupon_barrier=0.8,
            ki_barrier=0.6, coupon_rate=0.05, n_observations=4,
            T=1.0, r=0.03, sigma=0.20,
            vol_model=default_heston,
        )
        r1 = ac.price(n_paths=5_000, seed=42)
        r2 = ac.price(n_paths=50_000, seed=42)
        assert r2["std_error"] < r1["std_error"]
