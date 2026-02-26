"""Tests for implied volatility surface construction and SVI calibration."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vol_surface import (
    bs_call_price, bs_vega, implied_vol,
    svi_total_variance, svi_implied_vol,
    calibrate_svi_slice, svi_dw_dk, svi_d2w_dk2,
    ImpliedVolSurface,
)


class TestBlackScholes:
    def test_bs_call_atm(self):
        """ATM call price should be approximately S * N(d1) - K*exp(-rT)*N(d2)."""
        price = bs_call_price(100, 100, 1.0, 0.05, 0.20)
        assert 5 < price < 15  # reasonable range for ATM 1y call

    def test_bs_call_intrinsic(self):
        """Deep ITM call should be close to intrinsic."""
        price = bs_call_price(150, 50, 1.0, 0.05, 0.01)
        intrinsic = 150 - 50 * np.exp(-0.05)
        assert abs(price - intrinsic) < 1.0

    def test_bs_vega_positive(self):
        """Vega is always positive."""
        v = bs_vega(100, 100, 1.0, 0.05, 0.20)
        assert v > 0

    def test_bs_vega_atm_highest(self):
        """Vega is highest ATM."""
        v_atm = bs_vega(100, 100, 1.0, 0.05, 0.20)
        v_otm = bs_vega(100, 130, 1.0, 0.05, 0.20)
        assert v_atm > v_otm


class TestImpliedVol:
    def test_round_trip_atm(self):
        """BS price -> implied vol should recover original sigma."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r)
        assert iv is not None
        assert abs(iv - sigma) < 1e-6

    def test_round_trip_otm(self):
        """IV round-trip for OTM option."""
        S, K, T, r, sigma = 100, 120, 0.5, 0.03, 0.30
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r)
        assert iv is not None
        assert abs(iv - sigma) < 1e-5

    def test_round_trip_multiple_vols(self):
        """IV round-trip across different volatilities."""
        for sigma in [0.05, 0.10, 0.20, 0.50, 1.0]:
            price = bs_call_price(100, 100, 1.0, 0.03, sigma)
            iv = implied_vol(price, 100, 100, 1.0, 0.03)
            assert iv is not None
            assert abs(iv - sigma) < 1e-4, f"Failed for sigma={sigma}"

    def test_below_intrinsic_returns_none(self):
        """Price below intrinsic should return None."""
        iv = implied_vol(0.001, 100, 100, 1.0, 0.05)
        assert iv is None


class TestSVI:
    @pytest.fixture
    def typical_params(self):
        return (0.04, 0.08, -0.4, 0.01, 0.15)

    def test_svi_atm_value(self, typical_params):
        """SVI at k=0 should return finite positive total variance."""
        w = svi_total_variance(np.array([0.0]), typical_params)
        assert w[0] > 0
        assert np.isfinite(w[0])

    def test_svi_symmetric_for_zero_rho(self):
        """With rho=0, SVI should be symmetric around m."""
        params = (0.04, 0.1, 0.0, 0.0, 0.1)
        k_pos = svi_total_variance(np.array([0.1]), params)
        k_neg = svi_total_variance(np.array([-0.1]), params)
        assert abs(k_pos[0] - k_neg[0]) < 1e-10

    def test_svi_derivatives(self, typical_params):
        """SVI analytical derivatives should match finite differences."""
        k = 0.05
        dk = 1e-5
        w_up = svi_total_variance(np.array([k + dk]), typical_params)[0]
        w_down = svi_total_variance(np.array([k - dk]), typical_params)[0]

        # dw/dk
        fd_deriv1 = (w_up - w_down) / (2 * dk)
        analytical_deriv1 = svi_dw_dk(k, typical_params)
        assert abs(fd_deriv1 - analytical_deriv1) < 1e-4

        # d^2w/dk^2
        w_mid = svi_total_variance(np.array([k]), typical_params)[0]
        fd_deriv2 = (w_up - 2 * w_mid + w_down) / (dk**2)
        analytical_deriv2 = svi_d2w_dk2(k, typical_params)
        assert abs(fd_deriv2 - analytical_deriv2) < 1e-2

    def test_svi_butterfly_condition(self, typical_params):
        """d^2w/dk^2 >= 0 everywhere (butterfly no-arbitrage)."""
        for k in np.linspace(-0.5, 0.5, 50):
            d2w = svi_d2w_dk2(k, typical_params)
            assert d2w >= 0, f"Butterfly violated at k={k}: d2w/dk2={d2w}"


class TestSVICalibration:
    def test_recovers_known_params(self):
        """SVI calibration should approximately recover known parameters."""
        true_params = (0.04, 0.08, -0.4, 0.0, 0.15)
        k = np.linspace(-0.2, 0.2, 20)
        w = svi_total_variance(k, true_params)

        result = calibrate_svi_slice(k, w, T=1.0)
        assert result["success"]
        assert result["rmse"] < 0.001

    def test_handles_noisy_data(self):
        """SVI calibration should work with realistic noise."""
        rng = np.random.default_rng(42)
        true_params = (0.04, 0.08, -0.4, 0.0, 0.15)
        k = np.linspace(-0.2, 0.2, 15)
        w = svi_total_variance(k, true_params) + rng.normal(0, 0.001, len(k))

        result = calibrate_svi_slice(k, w, T=1.0)
        assert result["success"]
        assert result["rmse"] < 0.01

    def test_insufficient_points_fails(self):
        """SVI with < 5 points should fail gracefully."""
        k = np.array([0.0, 0.1, 0.2])
        w = np.array([0.04, 0.05, 0.06])
        result = calibrate_svi_slice(k, w, T=1.0)
        assert not result["success"]


class TestImpliedVolSurface:
    @pytest.fixture
    def mock_surface(self):
        """Build a surface from synthetic SVI params."""
        calibrations = {
            0.5: {"params": (0.02, 0.10, -0.3, 0.0, 0.1), "rmse": 0.001, "success": True, "n_points": 10},
            1.0: {"params": (0.04, 0.08, -0.4, 0.0, 0.15), "rmse": 0.001, "success": True, "n_points": 10},
            2.0: {"params": (0.08, 0.06, -0.5, 0.0, 0.2), "rmse": 0.001, "success": True, "n_points": 10},
        }
        return ImpliedVolSurface(spot=100, r=0.03, svi_calibrations=calibrations)

    def test_atm_vol_positive(self, mock_surface):
        for T in [0.5, 1.0, 1.5, 2.0]:
            v = mock_surface.atm_vol(T)
            assert v > 0, f"ATM vol negative at T={T}"
            assert v < 2.0, f"ATM vol unreasonably high at T={T}"

    def test_smile_shape(self, mock_surface):
        """OTM puts should have higher IV than ATM (equity skew)."""
        iv_otm_put = mock_surface.implied_vol_at(80, 1.0)
        iv_atm = mock_surface.implied_vol_at(100 * np.exp(0.03), 1.0)
        assert iv_otm_put > iv_atm

    def test_interpolation_between_maturities(self, mock_surface):
        """Interpolated vol at T=0.75 should be between T=0.5 and T=1.0."""
        K = 100
        v_05 = mock_surface.implied_vol_at(K, 0.5)
        v_10 = mock_surface.implied_vol_at(K, 1.0)
        v_075 = mock_surface.implied_vol_at(K, 0.75)
        assert min(v_05, v_10) * 0.9 < v_075 < max(v_05, v_10) * 1.1

    def test_shift_changes_atm(self, mock_surface):
        """Parallel shift should change ATM vol by approximately delta."""
        delta = 0.02
        shifted = mock_surface.shift(delta)
        for T in [0.5, 1.0, 2.0]:
            v_orig = mock_surface.atm_vol(T)
            v_shift = shifted.atm_vol(T)
            assert abs(v_shift - v_orig - delta) < 0.005


class TestFitQuality:
    @pytest.fixture
    def mock_surface(self):
        """Build a surface from synthetic SVI params."""
        calibrations = {
            0.5: {"params": (0.02, 0.10, -0.3, 0.0, 0.1), "rmse": 0.001, "success": True, "n_points": 10},
            1.0: {"params": (0.04, 0.08, -0.4, 0.0, 0.15), "rmse": 0.002, "success": True, "n_points": 10},
            2.0: {"params": (0.08, 0.06, -0.5, 0.0, 0.2), "rmse": 0.003, "success": True, "n_points": 10},
        }
        return ImpliedVolSurface(spot=100, r=0.03, svi_calibrations=calibrations)

    def test_fit_quality_returns_dict(self, mock_surface):
        result = mock_surface.fit_quality()
        assert isinstance(result, dict)

    def test_fit_quality_has_expected_keys(self, mock_surface):
        result = mock_surface.fit_quality()
        assert "overall_rmse" in result
        assert "per_slice_rmse" in result
        assert "per_slice_max_error" in result
        assert "n_slices" in result

    def test_fit_quality_n_slices(self, mock_surface):
        result = mock_surface.fit_quality()
        assert result["n_slices"] == 3

    def test_fit_quality_rmse_non_negative(self, mock_surface):
        result = mock_surface.fit_quality()
        assert result["overall_rmse"] >= 0
        for T, rmse in result["per_slice_rmse"].items():
            assert rmse >= 0


class TestArbitrageCheck:
    @pytest.fixture
    def mock_surface(self):
        """Build surface that should pass basic arbitrage checks."""
        calibrations = {
            0.5: {"params": (0.02, 0.10, -0.3, 0.0, 0.1), "rmse": 0.001, "success": True, "n_points": 10},
            1.0: {"params": (0.04, 0.08, -0.4, 0.0, 0.15), "rmse": 0.001, "success": True, "n_points": 10},
            2.0: {"params": (0.08, 0.06, -0.5, 0.0, 0.2), "rmse": 0.001, "success": True, "n_points": 10},
        }
        return ImpliedVolSurface(spot=100, r=0.03, svi_calibrations=calibrations)

    def test_arbitrage_check_returns_dict(self, mock_surface):
        result = mock_surface.arbitrage_check()
        assert isinstance(result, dict)

    def test_arbitrage_check_has_expected_keys(self, mock_surface):
        result = mock_surface.arbitrage_check()
        assert "butterfly_pass" in result
        assert "calendar_spread_pass" in result

    def test_butterfly_check_structure(self, mock_surface):
        result = mock_surface.arbitrage_check()
        assert isinstance(result["butterfly_pass"], bool)
        assert "butterfly_details" in result

    def test_calendar_spread_check_structure(self, mock_surface):
        result = mock_surface.arbitrage_check()
        assert isinstance(result["calendar_spread_pass"], bool)
        assert "calendar_details" in result
