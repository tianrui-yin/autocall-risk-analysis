"""Tests for Monte Carlo convergence analysis module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.convergence import convergence_analysis, convergence_report
from src.autocall import Autocallable


@pytest.fixture
def simple_autocall():
    """Simple autocallable for convergence tests (short maturity for speed)."""
    return Autocallable(
        S0=100, autocall_barrier=1.0, coupon_barrier=0.8,
        ki_barrier=0.6, coupon_rate=0.05, n_observations=4,
        T=1.0, r=0.03, sigma=0.25,
    )


class TestConvergenceAnalysis:
    def test_returns_correct_columns(self, simple_autocall):
        """Result should have all expected columns."""
        df = convergence_analysis(
            simple_autocall, path_counts=[1000, 5000, 10000], seed=42,
        )
        expected_cols = {"n_paths", "price", "std_error", "ci_lower", "ci_upper", "ci_width"}
        assert expected_cols.issubset(set(df.columns))

    def test_returns_correct_number_of_rows(self, simple_autocall):
        path_counts = [1000, 5000, 10000]
        df = convergence_analysis(
            simple_autocall, path_counts=path_counts, seed=42,
        )
        assert len(df) == len(path_counts)

    def test_ci_narrows_with_more_paths(self, simple_autocall):
        """Confidence interval should narrow as n_paths increases."""
        df = convergence_analysis(
            simple_autocall, path_counts=[5000, 50000], seed=42,
        )
        ci_widths = df["ci_width"].values
        assert ci_widths[1] < ci_widths[0]

    def test_std_error_decreases(self, simple_autocall):
        """Standard error should decrease with more paths."""
        df = convergence_analysis(
            simple_autocall, path_counts=[5000, 50000], seed=42,
        )
        se = df["std_error"].values
        assert se[1] < se[0]

    def test_approximate_sqrt_n_convergence(self, simple_autocall):
        """Std error ratio should approximately follow 1/sqrt(n) rate."""
        n1, n2 = 10000, 100000
        df = convergence_analysis(
            simple_autocall, path_counts=[n1, n2], seed=42,
        )
        se = df["std_error"].values
        # se ratio should be approximately sqrt(n2/n1) = sqrt(10) ≈ 3.16
        ratio = se[0] / se[1]
        expected_ratio = np.sqrt(n2 / n1)
        # Allow 50% tolerance due to MC noise
        assert 0.5 * expected_ratio < ratio < 2.0 * expected_ratio

    def test_ci_contains_price(self, simple_autocall):
        """Price should be within CI bounds."""
        df = convergence_analysis(
            simple_autocall, path_counts=[50000], seed=42,
        )
        row = df.iloc[0]
        assert row["ci_lower"] <= row["price"] <= row["ci_upper"]

    def test_prices_positive(self, simple_autocall):
        df = convergence_analysis(
            simple_autocall, path_counts=[5000, 20000], seed=42,
        )
        assert all(df["price"] > 0)


class TestConvergenceReport:
    def test_report_is_string(self, simple_autocall):
        report = convergence_report(
            simple_autocall, path_counts=[1000, 5000], seed=42,
        )
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_contains_key_info(self, simple_autocall):
        report = convergence_report(
            simple_autocall, path_counts=[1000, 5000], seed=42,
        )
        # Should contain path counts and price info
        assert "1000" in report or "1,000" in report
        assert "5000" in report or "5,000" in report
