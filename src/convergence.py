"""
Monte Carlo convergence analysis for autocallable pricing.

Analyzes how MC price estimates stabilize as the number of simulation paths
increases. Verifies the theoretical 1/sqrt(n) convergence rate and provides
confidence interval diagnostics.

Reference: Glasserman, "Monte Carlo Methods in Financial Engineering", Ch. 2.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .autocall import Autocallable


def convergence_analysis(
    product: Autocallable,
    path_counts: Optional[list[int]] = None,
    n_steps_per_period: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run MC pricing at increasing path counts and record convergence metrics.

    Parameters
    ----------
    product : Autocallable
        The product to price.
    path_counts : list of int, optional
        Number of paths for each run.
        Default: [10000, 50000, 100000, 200000, 500000].
    n_steps_per_period : int
        Time steps per observation period.
    seed : int
        Base random seed (different seed per run for independence).

    Returns
    -------
    pd.DataFrame
        Columns: n_paths, price, std_error, ci_lower, ci_upper, ci_width.
    """
    if path_counts is None:
        path_counts = [10_000, 50_000, 100_000, 200_000, 500_000]

    z_95 = 1.96  # 95% CI z-score

    records = []
    for n_paths in path_counts:
        result = product.price(
            n_paths=n_paths,
            n_steps_per_period=n_steps_per_period,
            seed=seed,
        )

        price = result["price"]
        se = result["std_error"]
        ci_lower = price - z_95 * se
        ci_upper = price + z_95 * se
        ci_width = 2 * z_95 * se

        records.append({
            "n_paths": n_paths,
            "price": price,
            "std_error": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_width,
        })

    return pd.DataFrame(records)


def convergence_report(
    product: Autocallable,
    path_counts: Optional[list[int]] = None,
    n_steps_per_period: int = 50,
    seed: int = 42,
) -> str:
    """
    Generate a formatted convergence analysis report.

    Parameters
    ----------
    product : Autocallable
        The product to price.
    path_counts : list of int, optional
        Number of paths for each run.
    n_steps_per_period : int
        Time steps per observation period.
    seed : int
        Base random seed.

    Returns
    -------
    str
        Formatted report with convergence table and rate analysis.
    """
    df = convergence_analysis(product, path_counts, n_steps_per_period, seed)

    lines = []
    lines.append("=" * 72)
    lines.append("MONTE CARLO CONVERGENCE ANALYSIS")
    lines.append("=" * 72)
    lines.append("")

    # Table header
    header = (
        f"{'n_paths':>10s}  {'Price':>10s}  {'Std Error':>10s}  "
        f"{'95% CI':>22s}  {'CI Width':>10s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in df.iterrows():
        n = int(row["n_paths"])
        lines.append(
            f"{n:>10,d}  {row['price']:>10.4f}  {row['std_error']:>10.4f}  "
            f"[{row['ci_lower']:>9.4f}, {row['ci_upper']:>9.4f}]  "
            f"{row['ci_width']:>10.4f}"
        )

    # Convergence rate analysis
    if len(df) >= 2:
        lines.append("")
        lines.append("-" * 72)
        lines.append("CONVERGENCE RATE ANALYSIS (1/sqrt(n) expected)")
        lines.append("-" * 72)

        se_vals = df["std_error"].values
        n_vals = df["n_paths"].values

        for i in range(1, len(df)):
            ratio_n = n_vals[i] / n_vals[i - 1]
            if se_vals[i] > 0:
                ratio_se = se_vals[i - 1] / se_vals[i]
            else:
                ratio_se = float("inf")
            expected_ratio = np.sqrt(ratio_n)
            lines.append(
                f"  {int(n_vals[i-1]):>10,d} -> {int(n_vals[i]):>10,d}: "
                f"SE ratio = {ratio_se:.3f}  (expected {expected_ratio:.3f})"
            )

    lines.append("")
    lines.append(f"Final estimate: {df.iloc[-1]['price']:.4f} "
                 f"+/- {df.iloc[-1]['std_error']:.4f}")
    lines.append("=" * 72)

    return "\n".join(lines)
