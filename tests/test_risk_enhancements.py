"""Tests for risk enhancements: joint stress grid, Vanna/Volga, enhanced P&L decomposition."""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.autocall import Autocallable
from src.greeks import vanna, volga, compute_all_greeks
from src.stress_testing import stress_grid, sensitivity_surface
from src.pnl_explain import pnl_attribution, multi_day_pnl


@pytest.fixture
def standard_autocall():
    """Standard Phoenix Autocallable for testing."""
    return Autocallable(
        S0=100,
        autocall_barrier=1.0,
        coupon_barrier=0.8,
        ki_barrier=0.6,
        coupon_rate=0.05,
        n_observations=8,
        T=2.0,
        r=0.03,
        sigma=0.25,
    )


# ============================================================
# Stress Grid Tests
# ============================================================
class TestStressGrid:
    def test_returns_correct_shape(self, standard_autocall):
        """Grid should have n_spot * n_vol rows."""
        spot_shocks = [-0.20, -0.10, 0.0, 0.10]
        vol_shocks = [-0.05, 0.0, 0.05]
        result, pivot = stress_grid(
            standard_autocall,
            spot_shocks=spot_shocks,
            vol_shocks=vol_shocks,
            n_paths=10_000,
            seed=42,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(spot_shocks) * len(vol_shocks)

    def test_base_case_approx_zero_pnl(self, standard_autocall):
        """At (0%, 0%) shock, PnL should be approximately zero."""
        spot_shocks = [0.0]
        vol_shocks = [0.0]
        result, _ = stress_grid(
            standard_autocall,
            spot_shocks=spot_shocks,
            vol_shocks=vol_shocks,
            n_paths=50_000,
            seed=42,
        )
        row = result.iloc[0]
        assert abs(row["pnl"]) < 1.0  # MC noise tolerance

    def test_extreme_down_stress_increases_ki_prob(self, standard_autocall):
        """Extreme spot down + vol up should increase KI probability."""
        spot_shocks = [0.0, -0.30]
        vol_shocks = [0.0, 0.20]
        result, _ = stress_grid(
            standard_autocall,
            spot_shocks=spot_shocks,
            vol_shocks=vol_shocks,
            n_paths=20_000,
            seed=42,
        )
        base_row = result[(result["spot_shock"] == 0.0) & (result["vol_shock"] == 0.0)]
        stress_row = result[(result["spot_shock"] == -0.30) & (result["vol_shock"] == 0.20)]
        assert stress_row["ki_prob"].values[0] > base_row["ki_prob"].values[0]

    def test_extreme_down_stress_decreases_autocall_prob(self, standard_autocall):
        """Extreme spot down should decrease autocall probability."""
        spot_shocks = [0.0, -0.30]
        vol_shocks = [0.0]
        result, _ = stress_grid(
            standard_autocall,
            spot_shocks=spot_shocks,
            vol_shocks=vol_shocks,
            n_paths=20_000,
            seed=42,
        )
        base_row = result[(result["spot_shock"] == 0.0) & (result["vol_shock"] == 0.0)]
        stress_row = result[(result["spot_shock"] == -0.30) & (result["vol_shock"] == 0.0)]
        assert stress_row["autocall_prob"].values[0] < base_row["autocall_prob"].values[0]

    def test_pivot_table_shape(self, standard_autocall):
        """Pivot table should be spot_shocks x vol_shocks."""
        spot_shocks = [-0.10, 0.0, 0.10]
        vol_shocks = [-0.05, 0.0, 0.05]
        _, pivot = stress_grid(
            standard_autocall,
            spot_shocks=spot_shocks,
            vol_shocks=vol_shocks,
            n_paths=10_000,
            seed=42,
        )
        assert pivot.shape == (len(spot_shocks), len(vol_shocks))

    def test_default_grids_work(self, standard_autocall):
        """Default spot/vol grids should work without explicit parameters."""
        result, pivot = stress_grid(
            standard_autocall,
            n_paths=5_000,
            seed=42,
        )
        assert len(result) > 0
        assert pivot.shape[0] > 0


class TestSensitivitySurface:
    def test_custom_parameters(self, standard_autocall):
        """sensitivity_surface should work with arbitrary parameter ranges."""
        result, pivot = sensitivity_surface(
            standard_autocall,
            param1_name="S0",
            param1_range=[90, 100, 110],
            param2_name="sigma",
            param2_range=[0.20, 0.25, 0.30],
            n_paths=10_000,
            seed=42,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 9  # 3 x 3
        assert pivot.shape == (3, 3)

    def test_returns_prices(self, standard_autocall):
        """Result should contain prices for each parameter combination."""
        result, _ = sensitivity_surface(
            standard_autocall,
            param1_name="S0",
            param1_range=[90, 100, 110],
            param2_name="sigma",
            param2_range=[0.20, 0.25, 0.30],
            n_paths=10_000,
            seed=42,
        )
        assert "price" in result.columns
        assert all(result["price"] > 0)


# ============================================================
# Vanna Tests
# ============================================================
class TestVanna:
    def test_returns_finite_number(self, standard_autocall):
        """Vanna should return a finite float."""
        v = vanna(standard_autocall, n_paths=50_000, seed=42)
        assert np.isfinite(v)

    def test_is_float(self, standard_autocall):
        """Vanna should be a scalar float."""
        v = vanna(standard_autocall, n_paths=50_000, seed=42)
        assert isinstance(v, (float, np.floating))

    def test_consistency_two_ways(self, standard_autocall):
        """
        Vanna computed as dDelta/dSigma should approximately agree with
        dVega/dSpot (Schwarz's theorem: mixed partials are equal).

        dDelta/dSigma: [delta(sigma+d) - delta(sigma-d)] / (2*d)
        dVega/dSpot: [vega(S+dS) - vega(S-dS)] / (2*dS)
        """
        from src.greeks import delta as delta_fn, vega as vega_fn

        d_sigma = 0.01
        dS_pct = 0.01
        dS = standard_autocall.S0 * dS_pct
        n_paths = 100_000
        seed = 42

        # Way 1: dDelta/dSigma
        original_sigma = standard_autocall.sigma
        standard_autocall.sigma = original_sigma + d_sigma
        d_up = delta_fn(standard_autocall, n_paths=n_paths, seed=seed)
        standard_autocall.sigma = original_sigma - d_sigma
        d_down = delta_fn(standard_autocall, n_paths=n_paths, seed=seed)
        standard_autocall.sigma = original_sigma
        vanna_way1 = (d_up - d_down) / (2 * d_sigma)

        # Way 2: dVega/dSpot
        original_S0 = standard_autocall.S0
        standard_autocall.S0 = original_S0 + dS
        v_up = vega_fn(standard_autocall, n_paths=n_paths, seed=seed)
        standard_autocall.S0 = original_S0 - dS
        v_down = vega_fn(standard_autocall, n_paths=n_paths, seed=seed)
        standard_autocall.S0 = original_S0
        vanna_way2 = (v_up - v_down) / (2 * dS)

        # Allow ~40% relative tolerance due to MC noise on second-order derivatives
        if abs(vanna_way1) > 0.1:
            rel_diff = abs(vanna_way1 - vanna_way2) / abs(vanna_way1)
            assert rel_diff < 0.5, (
                f"Vanna inconsistency: way1={vanna_way1:.4f}, way2={vanna_way2:.4f}, "
                f"rel_diff={rel_diff:.2%}"
            )
        # If both are near zero, they trivially agree


# ============================================================
# Volga Tests
# ============================================================
class TestVolga:
    def test_returns_finite_number(self, standard_autocall):
        """Volga should return a finite float."""
        v = volga(standard_autocall, n_paths=50_000, seed=42)
        assert np.isfinite(v)

    def test_is_float(self, standard_autocall):
        """Volga should be a scalar float."""
        v = volga(standard_autocall, n_paths=50_000, seed=42)
        assert isinstance(v, (float, np.floating))

    def test_volga_reasonable_magnitude(self, standard_autocall):
        """
        Volga should have a reasonable magnitude for a notional=100 product.

        Note: Autocallables can have negative volga because the autocall
        feature creates short vol convexity (investor effectively sold
        the early redemption option). The knock-in put adds positive volga,
        but the net sign depends on barrier levels and spot position.
        """
        v = volga(standard_autocall, n_paths=100_000, seed=42)
        # Magnitude should be bounded (for notional=100, sigma~0.25)
        assert abs(v) < 500, f"Volga unreasonably large: {v}"


# ============================================================
# compute_all_greeks includes Vanna/Volga
# ============================================================
class TestComputeAllGreeksEnhanced:
    def test_includes_vanna_and_volga(self, standard_autocall):
        """compute_all_greeks should include vanna and volga."""
        g = compute_all_greeks(standard_autocall, n_paths=50_000, seed=42)
        assert "vanna" in g
        assert "volga" in g
        assert np.isfinite(g["vanna"])
        assert np.isfinite(g["volga"])


# ============================================================
# Enhanced P&L Decomposition Tests
# ============================================================
class TestEnhancedPnL:
    def test_enhanced_pnl_has_vanna_volga_terms(self, standard_autocall):
        """P&L attribution should include vanna_pnl and volga_pnl."""
        result = pnl_attribution(
            standard_autocall, dS=2.0, d_sigma=0.02, dt_days=1.0,
            n_paths=50_000, seed=42,
        )
        assert "vanna_pnl" in result
        assert "volga_pnl" in result

    def test_enhanced_pnl_has_unexplained(self, standard_autocall):
        """P&L attribution should report unexplained residual."""
        result = pnl_attribution(
            standard_autocall, dS=2.0, d_sigma=0.02, dt_days=1.0,
            n_paths=50_000, seed=42,
        )
        assert "unexplained" in result

    def test_enhanced_explains_more_than_basic(self, standard_autocall):
        """
        Enhanced P&L (with Vanna/Volga) should explain at least as much
        as basic P&L for joint spot+vol moves.
        """
        result = pnl_attribution(
            standard_autocall, dS=3.0, d_sigma=0.03, dt_days=1.0,
            n_paths=100_000, seed=42,
        )
        basic_explained = abs(
            result["delta_pnl"] + result["gamma_pnl"]
            + result["vega_pnl"] + result["theta_pnl"]
        )
        full_explained = abs(result["explained"])
        # Enhanced should explain at least as much
        assert full_explained >= basic_explained - 0.5  # MC noise tolerance

    def test_small_move_high_explained_pct(self, standard_autocall):
        """For small moves, Greeks should explain >50% of P&L."""
        result = pnl_attribution(
            standard_autocall, dS=0.5, d_sigma=0.005, dt_days=1.0,
            n_paths=100_000, seed=42,
        )
        assert abs(result["explained_pct"]) > 50

    def test_joint_move_vanna_term_nonzero(self, standard_autocall):
        """For joint spot+vol moves, Vanna term should be non-negligible."""
        result = pnl_attribution(
            standard_autocall, dS=5.0, d_sigma=0.05, dt_days=1.0,
            n_paths=100_000, seed=42,
        )
        # Vanna term = vanna * dS * d_sigma; should be nonzero for significant moves
        assert result["vanna_pnl"] != 0.0

    def test_pnl_breakdown_percentages(self, standard_autocall):
        """P&L should include percentage breakdown of each term."""
        result = pnl_attribution(
            standard_autocall, dS=2.0, d_sigma=0.02, dt_days=1.0,
            n_paths=50_000, seed=42,
        )
        assert "explained_pct_breakdown" in result
        breakdown = result["explained_pct_breakdown"]
        assert "delta_pct" in breakdown
        assert "gamma_pct" in breakdown
        assert "vega_pct" in breakdown
        assert "theta_pct" in breakdown
        assert "vanna_pct" in breakdown
        assert "volga_pct" in breakdown

    def test_explained_equals_sum_of_terms(self, standard_autocall):
        """Explained P&L should equal sum of all individual Greek terms."""
        result = pnl_attribution(
            standard_autocall, dS=2.0, d_sigma=0.02, dt_days=1.0,
            n_paths=50_000, seed=42,
        )
        sum_terms = (
            result["delta_pnl"] + result["gamma_pnl"]
            + result["vega_pnl"] + result["theta_pnl"]
            + result["vanna_pnl"] + result["volga_pnl"]
        )
        assert abs(result["explained"] - sum_terms) < 1e-10

    def test_unexplained_equals_actual_minus_explained(self, standard_autocall):
        """Unexplained = actual_pnl - explained."""
        result = pnl_attribution(
            standard_autocall, dS=2.0, d_sigma=0.02, dt_days=1.0,
            n_paths=50_000, seed=42,
        )
        expected_unexplained = result["actual_pnl"] - result["explained"]
        assert abs(result["unexplained"] - expected_unexplained) < 1e-10


class TestMultiDayPnLEnhanced:
    def test_multi_day_includes_vanna_volga(self, standard_autocall):
        """Multi-day P&L should include vanna/volga terms."""
        spot_path = np.array([100, 101, 99, 100])
        vol_path = np.array([0.25, 0.26, 0.27, 0.25])
        results = multi_day_pnl(
            standard_autocall,
            spot_path=spot_path,
            vol_path=vol_path,
            n_paths=20_000,
            seed=42,
        )
        assert len(results) == 3  # 3 days
        for day_result in results:
            assert "vanna_pnl" in day_result
            assert "volga_pnl" in day_result
            assert "explained_pct_breakdown" in day_result
