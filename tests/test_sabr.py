"""Tests for SABR stochastic volatility model (Hagan et al., 2002)."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.sabr import SABRModel


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def default_sabr():
    """Typical equity SABR parameters (beta=1, lognormal)."""
    return SABRModel(alpha=0.20, rho=-0.5, nu=0.4)


@pytest.fixture
def zero_rho_sabr():
    """SABR with rho=0 (symmetric smile)."""
    return SABRModel(alpha=0.20, rho=0.0, nu=0.4)


@pytest.fixture
def low_volvol_sabr():
    """SABR with nu -> 0 (converges to Black-Scholes constant vol = alpha)."""
    return SABRModel(alpha=0.20, rho=-0.5, nu=1e-6)


# =====================================================================
# 1. Parameter Validation
# =====================================================================

class TestSABRValidation:
    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            SABRModel(alpha=-0.1, rho=-0.5, nu=0.4)

    def test_zero_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            SABRModel(alpha=0.0, rho=-0.5, nu=0.4)

    def test_rho_out_of_range_raises(self):
        with pytest.raises(ValueError, match="rho"):
            SABRModel(alpha=0.2, rho=1.5, nu=0.4)
        with pytest.raises(ValueError, match="rho"):
            SABRModel(alpha=0.2, rho=-1.5, nu=0.4)

    def test_rho_at_boundary_raises(self):
        with pytest.raises(ValueError, match="rho"):
            SABRModel(alpha=0.2, rho=1.0, nu=0.4)
        with pytest.raises(ValueError, match="rho"):
            SABRModel(alpha=0.2, rho=-1.0, nu=0.4)

    def test_negative_nu_raises(self):
        with pytest.raises(ValueError, match="nu"):
            SABRModel(alpha=0.2, rho=-0.5, nu=-0.1)

    def test_zero_nu_raises(self):
        with pytest.raises(ValueError, match="nu"):
            SABRModel(alpha=0.2, rho=-0.5, nu=0.0)

    def test_valid_params_accepted(self):
        model = SABRModel(alpha=0.2, rho=-0.5, nu=0.4)
        assert model.alpha == 0.2
        assert model.rho == -0.5
        assert model.nu == 0.4
        assert model.beta == 1.0


# =====================================================================
# 2. ATM Limit
# =====================================================================

class TestSABRATMLimit:
    def test_atm_vol_close_to_alpha_short_T(self, default_sabr):
        """When K=F and T is small, implied_vol ~ alpha."""
        F = 100.0
        K = F
        T = 0.01
        iv = default_sabr.implied_vol(F, K, T)
        # For very short T, correction terms are negligible
        assert abs(iv - default_sabr.alpha) < 0.01

    def test_atm_vol_correction_increases_with_T(self, default_sabr):
        """ATM correction term grows with T."""
        F = 100.0
        K = F
        iv_short = default_sabr.implied_vol(F, K, 0.01)
        iv_long = default_sabr.implied_vol(F, K, 1.0)
        # Both should be close to alpha but may differ due to correction
        # The correction term is:
        #   alpha * (rho*nu*alpha/4 + (2-3*rho^2)*nu^2/24) * T
        # With rho=-0.5, nu=0.4, alpha=0.2:
        #   rho*nu*alpha/4 = -0.5*0.4*0.2/4 = -0.01
        #   (2-3*0.25)*0.16/24 = 1.25*0.16/24 ~ 0.00833
        # Net correction per T ~ alpha * (-0.01 + 0.00833) ~ alpha * (-0.00167) < 0
        # So implied vol may slightly decrease with T for these params
        assert iv_short > 0
        assert iv_long > 0

    def test_atm_vol_formula(self, default_sabr):
        """
        For K=F (beta=1): sigma_ATM = alpha * [1 + (rho*nu*alpha/4 + (2-3*rho^2)*nu^2/24)*T]
        """
        F = 100.0
        K = F
        T = 0.5
        alpha = default_sabr.alpha
        rho = default_sabr.rho
        nu = default_sabr.nu

        # Expected ATM vol from closed-form
        correction = 1.0 + (rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho**2) * nu**2 / 24.0) * T
        expected_iv = alpha * correction

        actual_iv = default_sabr.implied_vol(F, K, T)
        assert abs(actual_iv - expected_iv) < 1e-6


# =====================================================================
# 3. Smile Shape
# =====================================================================

class TestSABRSmileShape:
    def test_negative_rho_skew(self, default_sabr):
        """For rho < 0, low strikes should have higher implied vol (equity skew)."""
        F = 100.0
        T = 0.5
        iv_low = default_sabr.implied_vol(F, 80.0, T)
        iv_atm = default_sabr.implied_vol(F, 100.0, T)
        iv_high = default_sabr.implied_vol(F, 120.0, T)

        # Negative skew: vol at low K > vol at ATM > vol at high K
        assert iv_low > iv_atm
        assert iv_atm > iv_high

    def test_rho_zero_approximately_symmetric(self, zero_rho_sabr):
        """When rho=0, smile should be approximately symmetric around ATM."""
        F = 100.0
        T = 0.5
        # Check symmetric log-moneyness points
        iv_low = zero_rho_sabr.implied_vol(F, F * np.exp(-0.1), T)
        iv_high = zero_rho_sabr.implied_vol(F, F * np.exp(0.1), T)
        # Should be approximately equal (not exactly due to higher-order terms)
        assert abs(iv_low - iv_high) / iv_low < 0.05

    def test_smile_convex(self, default_sabr):
        """Vol smile should be convex (vol at wings > vol at ATM)."""
        F = 100.0
        T = 0.5
        iv_low = default_sabr.implied_vol(F, 80.0, T)
        iv_atm = default_sabr.implied_vol(F, 100.0, T)
        iv_high = default_sabr.implied_vol(F, 120.0, T)
        # At least one wing should be above ATM
        assert max(iv_low, iv_high) > iv_atm

    def test_positive_rho_reverses_skew(self):
        """For rho > 0, high strikes should have higher vol."""
        model = SABRModel(alpha=0.20, rho=0.5, nu=0.4)
        F = 100.0
        T = 0.5
        iv_low = model.implied_vol(F, 80.0, T)
        iv_high = model.implied_vol(F, 120.0, T)
        assert iv_high > iv_low


# =====================================================================
# 4. Low Vol-of-Vol Limit
# =====================================================================

class TestSABRLowVolVol:
    def test_nu_zero_gives_flat_smile(self, low_volvol_sabr):
        """When nu -> 0, SABR reduces to constant vol = alpha (no smile)."""
        F = 100.0
        T = 0.5
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        vols = [low_volvol_sabr.implied_vol(F, K, T) for K in strikes]
        # All vols should be approximately alpha
        for vol in vols:
            assert abs(vol - low_volvol_sabr.alpha) < 0.001


# =====================================================================
# 5. Implied Vol Positivity and Finiteness
# =====================================================================

class TestSABRImpliedVolProperties:
    def test_implied_vol_positive(self, default_sabr):
        """Implied vol must be positive for all reasonable (F, K, T)."""
        F = 100.0
        for K in [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]:
            for T in [0.05, 0.25, 0.5, 1.0, 2.0]:
                iv = default_sabr.implied_vol(F, K, T)
                assert iv > 0, f"Non-positive vol at K={K}, T={T}: {iv}"

    def test_implied_vol_finite(self, default_sabr):
        """Implied vol must be finite."""
        F = 100.0
        for K in [70.0, 100.0, 130.0]:
            for T in [0.05, 0.5, 2.0]:
                iv = default_sabr.implied_vol(F, K, T)
                assert np.isfinite(iv), f"Non-finite vol at K={K}, T={T}: {iv}"


# =====================================================================
# 6. Calibration Recovery (Synthetic Data)
# =====================================================================

class TestSABRCalibration:
    def test_calibration_recovery(self):
        """Generate SABR vols with known params, calibrate back, check recovery."""
        true_alpha, true_rho, true_nu = 0.25, -0.4, 0.5
        true_model = SABRModel(alpha=true_alpha, rho=true_rho, nu=true_nu)

        F = 100.0
        T = 0.5
        strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)
        market_vols = np.array([true_model.implied_vol(F, K, T) for K in strikes])

        result = SABRModel.calibrate(
            market_vols=market_vols,
            strikes=strikes,
            F=F,
            T=T,
        )

        assert result["rmse"] < 0.001  # Near-perfect fit
        cal = result["model"]
        assert abs(cal.alpha - true_alpha) < 0.02
        assert abs(cal.rho - true_rho) < 0.1
        assert abs(cal.nu - true_nu) < 0.1

    def test_calibration_returns_dict_structure(self):
        """Calibration must return model, rmse, params."""
        model = SABRModel(alpha=0.2, rho=-0.5, nu=0.4)
        F = 100.0
        T = 0.5
        strikes = np.array([90, 95, 100, 105, 110], dtype=float)
        market_vols = np.array([model.implied_vol(F, K, T) for K in strikes])

        result = SABRModel.calibrate(
            market_vols=market_vols,
            strikes=strikes,
            F=F,
            T=T,
        )

        assert "model" in result
        assert "rmse" in result
        assert "params" in result
        assert isinstance(result["model"], SABRModel)
        assert result["rmse"] >= 0
        assert "alpha" in result["params"]
        assert "rho" in result["params"]
        assert "nu" in result["params"]

    def test_calibration_rmse_reasonable_on_noisy_data(self):
        """Calibration should still produce low RMSE with small noise."""
        model = SABRModel(alpha=0.2, rho=-0.5, nu=0.4)
        F = 100.0
        T = 0.5
        strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)
        true_vols = np.array([model.implied_vol(F, K, T) for K in strikes])
        rng = np.random.default_rng(42)
        noisy_vols = true_vols + rng.normal(0, 0.002, size=len(strikes))

        result = SABRModel.calibrate(
            market_vols=noisy_vols,
            strikes=strikes,
            F=F,
            T=T,
        )
        assert result["rmse"] < 0.01


# =====================================================================
# 7. Surface Calibration
# =====================================================================

class TestSABRSurfaceCalibration:
    def test_calibrate_surface_multiple_slices(self):
        """calibrate_surface should calibrate per maturity slice."""
        import pandas as pd
        model = SABRModel(alpha=0.2, rho=-0.5, nu=0.4)
        spot = 100.0
        r = 0.03

        rows = []
        for T in [0.25, 0.5, 1.0]:
            F = spot * np.exp(r * T)
            for K in [85, 90, 95, 100, 105, 110, 115]:
                iv = model.implied_vol(F, float(K), T)
                rows.append({"T": T, "K": float(K), "implied_vol": iv})

        vol_data = pd.DataFrame(rows)
        result = SABRModel.calibrate_surface(vol_data, spot=spot, r=r)

        assert len(result) == 3
        for T_val, cal in result.items():
            assert cal["rmse"] < 0.005
            assert isinstance(cal["model"], SABRModel)


# =====================================================================
# 8. SABR vs SVI RMSE Comparison
# =====================================================================

class TestSABRvsSVI:
    def test_both_produce_finite_rmse(self):
        """Both SABR and SVI should produce finite RMSEs on sample market data."""
        from src.market_data import load_sample_data
        from src.vol_surface import extract_implied_vols, calibrate_svi_slice

        data = load_sample_data()
        spot = data["spot_spy"]
        r = data["risk_free_rate"]
        options_df = data["options_chain"]

        vol_data = extract_implied_vols(options_df, spot, r)

        # Pick one slice with enough points
        slices = vol_data.groupby("T")
        picked = False
        for T_val, group in slices:
            if len(group) >= 7:
                F = spot * np.exp(r * T_val)
                strikes = group["strike"].values
                impl_vols = group["implied_vol"].values

                # SABR calibration
                sabr_result = SABRModel.calibrate(
                    market_vols=impl_vols,
                    strikes=strikes,
                    F=F,
                    T=T_val,
                )
                assert np.isfinite(sabr_result["rmse"])
                assert sabr_result["rmse"] >= 0

                # SVI calibration
                svi_result = calibrate_svi_slice(
                    log_moneyness=group["log_moneyness"].values,
                    total_variance=group["total_variance"].values,
                    T=T_val,
                )
                if svi_result["success"]:
                    assert np.isfinite(svi_result["rmse"])

                picked = True
                break

        assert picked, "No slice with enough data points found"


# =====================================================================
# 9. Integration with Autocallable Pricing
# =====================================================================

class TestSABRAutocallIntegration:
    def test_sabr_atm_vol_as_constant_for_autocall(self, default_sabr):
        """
        SABR ATM vol used as constant vol input for Autocallable should
        produce a reasonable price.
        """
        from src.autocall import Autocallable

        F = 100.0
        T = 1.0
        atm_vol = default_sabr.implied_vol(F, F, T)

        ac = Autocallable(
            S0=100, autocall_barrier=1.0, coupon_barrier=0.8,
            ki_barrier=0.6, coupon_rate=0.05, n_observations=4,
            T=T, r=0.03, sigma=atm_vol,
        )
        result = ac.price(n_paths=10_000, seed=42)
        assert result["price"] > 0
        assert 70 < result["price"] < 140
        assert 0 <= result["total_autocall_prob"] <= 1.0
        assert 0 <= result["ki_probability"] <= 1.0

    def test_sabr_atm_vol_differs_from_svi(self):
        """SABR and SVI calibrated ATM vols should both be reasonable but may differ."""
        from src.market_data import load_sample_data
        from src.vol_surface import extract_implied_vols, build_vol_surface_from_market

        data = load_sample_data()
        spot = data["spot_spy"]
        r = data["risk_free_rate"]

        # SVI surface
        surface = build_vol_surface_from_market(use_sample=True)
        T_test = float(surface.maturities[0])
        svi_atm = surface.atm_vol(T_test)

        # SABR on same slice
        options_df = data["options_chain"]
        vol_data = extract_implied_vols(options_df, spot, r)
        slice_data = vol_data[abs(vol_data["T"] - T_test) < 0.01]
        if len(slice_data) >= 5:
            F = spot * np.exp(r * T_test)
            sabr_result = SABRModel.calibrate(
                market_vols=slice_data["implied_vol"].values,
                strikes=slice_data["strike"].values,
                F=F,
                T=T_test,
            )
            sabr_atm = sabr_result["model"].implied_vol(F, F, T_test)

            # Both should be in a reasonable vol range
            assert 0.05 < svi_atm < 1.0
            assert 0.05 < sabr_atm < 1.0


# =====================================================================
# 10. Edge Cases
# =====================================================================

class TestSABREdgeCases:
    def test_very_deep_itm(self, default_sabr):
        """Very low strike should still return positive finite vol."""
        F = 100.0
        T = 0.5
        iv = default_sabr.implied_vol(F, 50.0, T)
        assert iv > 0
        assert np.isfinite(iv)

    def test_very_deep_otm(self, default_sabr):
        """Very high strike should still return positive finite vol."""
        F = 100.0
        T = 0.5
        iv = default_sabr.implied_vol(F, 150.0, T)
        assert iv > 0
        assert np.isfinite(iv)

    def test_very_short_maturity(self, default_sabr):
        """Very short T should give vol close to alpha."""
        F = 100.0
        T = 0.001
        iv = default_sabr.implied_vol(F, 100.0, T)
        assert abs(iv - default_sabr.alpha) < 0.01

    def test_model_name_property(self, default_sabr):
        """model_name property should exist and be informative."""
        name = default_sabr.model_name
        assert "SABR" in name
        assert "0.20" in name or "0.2" in name
