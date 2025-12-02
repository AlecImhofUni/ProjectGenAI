#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a wide (sigma, sigma_blur) table and extract a tagged Top-10 subset.

What this script does
---------------------
Inputs:
  - A "GLOBAL" sweep CSV (one row per (sigma, sigma_blur) with global AUROC).
  - A per-dataset sweep CSV (one row per (sigma, sigma_blur, dataset) with AUROC).

From these, it:
  1. Ensures both tables have numeric `sigma_blur` (either from a column or
     inferred from `results_dir` using the pattern: `...sblur_<val>...`).
  2. Aggregates per-dataset AUROC to keep the best value per
     (sigma, sigma_blur, dataset).
  3. Pivots to a wide table indexed by (sigma, sigma_blur) with:
       - One column per dataset: "<DATASET> AUROC"
       - Global AUROC column (and optional global_auroc_wo_sid if present).
     → Saved as: out_tables/table_by_sigma.csv
  4. Builds a "Top-10 union" table:
       - For each metric (global, per-dataset, and optional global_wo_sid),
         takes the Top-10 rows.
       - Unions all these rows and adds a `top10_for` column listing the
         metrics that selected each row (e.g. "global; ADM; CollabDiff").
     → Saved as: out_tables/top_union_table_by_sigma.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame, with a helpful error if missing.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        pandas.DataFrame: Parsed CSV contents.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_csv(path)


def ensure_sigma_blur(
    df: pd.DataFrame,
    sigma_blur_col: str,
    results_dir_col: str = "results_dir",
) -> pd.DataFrame:
    """
    Ensure the DataFrame has a float 'sigma_blur' column.

    If `sigma_blur_col` is missing, tries to infer it from `results_dir` using
    a regex pattern that matches substrings like:
        "...sblur_0.55" or "...sblur-0.55"

    Args:
        df (pandas.DataFrame): Input DataFrame.
        sigma_blur_col (str): Name of the sigma_blur column to enforce.
        results_dir_col (str): Column containing paths, used to infer
            sigma_blur if needed.

    Returns:
        pandas.DataFrame: The same DataFrame with a float `sigma_blur_col`.

    Raises:
        ValueError: If sigma_blur cannot be inferred from `results_dir_col`.
    """
    if sigma_blur_col not in df.columns:
        if results_dir_col not in df.columns:
            raise ValueError(f"Missing '{sigma_blur_col}' and '{results_dir_col}' to infer it.")
        # Extract the numeric part after "sblur_" or "sblur-"
        extr = df[results_dir_col].astype(str).str.extract(r"(?:^|[_/])sblur[_-]?([0-9.]+)")[0]
        if extr.isna().any():
            raise ValueError("Could not infer sigma_blur from results_dir; need '...sblur_<val>'.")
        df[sigma_blur_col] = extr.astype(float)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a wide (sigma, sigma_blur) table with global and per-dataset "
            "AUROCs, plus a Top-10 union table with tags indicating which metric "
            "selected each row."
        )
    )
    parser.add_argument(
        "--global-file",
        default="results_minder_dinov3/minder_sweep_global_no_sid.csv",
        help="Global sweep CSV (must contain global AUROC; may contain 'global_auroc_wo_sid').",
    )
    parser.add_argument(
        "--dataset-file",
        default="results_minder_dinov3/minder_sweep_per_dataset.csv",
        help="Per-dataset sweep CSV (one row per (sigma, sigma_blur, dataset)).",
    )
    parser.add_argument("--sigma-col", default="sigma", help="Column name for sigma.")
    parser.add_argument("--sigma-blur-col", default="sigma_blur", help="Column name for sigma_blur.")
    parser.add_argument(
        "--global-auroc-col",
        default="global_auroc",
        help="Column name for the main global AUROC.",
    )
    parser.add_argument(
        "--maybe-global-auroc-wo-sid",
        default="global_auroc_wo_sid",
        help="Optional extra global metric (e.g., without SID); used if present.",
    )
    parser.add_argument(
        "--dataset-col",
        default="dataset",
        help="Column name for dataset identifiers in the per-dataset CSV.",
    )
    parser.add_argument(
        "--dataset-auroc-col",
        default="auroc",
        help="Column name for per-dataset AUROC values.",
    )
    parser.add_argument(
        "--out-dir",
        default="out_tables",
        help="Output directory for the generated CSV tables.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=6,
        help="Number of decimal places to round AUROC values to.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- GLOBAL ---------- #
    g = load_csv(Path(args.global_file))

    # We at least need: sigma, results_dir, and the main global AUROC
    need_g = {args.sigma_col, "results_dir", args.global_auroc_col}
    if args.sigma_blur_col in g.columns:
        need_g.add(args.sigma_blur_col)
    miss = need_g - set(g.columns)
    if miss:
        raise ValueError(f"[Global] Missing columns {miss}. Found: {list(g.columns)}")

    # Ensure sigma_blur is present (either as column or inferred from results_dir)
    g = ensure_sigma_blur(g, args.sigma_blur_col, "results_dir")
    has_wo_sid = args.maybe_global_auroc_wo_sid in g.columns

    # Keep only the global metrics we care about
    keep_g_cols = [args.sigma_col, args.sigma_blur_col, args.global_auroc_col]
    if has_wo_sid:
        keep_g_cols.append(args.maybe_global_auroc_wo_sid)
    g = g[keep_g_cols].copy()
    g[args.sigma_col] = g[args.sigma_col].astype(float)
    g[args.sigma_blur_col] = g[args.sigma_blur_col].astype(float)

    # ---------- PER-DATASET ---------- #
    d = load_csv(Path(args.dataset_file))

    # Need sigma, dataset name, per-dataset AUROC, and results_dir
    need_d = {args.sigma_col, args.dataset_col, args.dataset_auroc_col, "results_dir"}
    if args.sigma_blur_col in d.columns:
        need_d.add(args.sigma_blur_col)
    miss = need_d - set(d.columns)
    if miss:
        raise ValueError(f"[Dataset] Missing columns {miss}. Found: {list(d.columns)}")

    d = ensure_sigma_blur(d, args.sigma_blur_col, "results_dir")
    d = d[[args.sigma_col, args.sigma_blur_col, args.dataset_col, args.dataset_auroc_col]].copy()
    d[args.sigma_col] = d[args.sigma_col].astype(float)
    d[args.sigma_blur_col] = d[args.sigma_blur_col].astype(float)

    # If there are duplicates per (sigma, sigma_blur, dataset), keep the best AUROC
    d_best = (
        d.groupby([args.sigma_col, args.sigma_blur_col, args.dataset_col], as_index=False)[
            args.dataset_auroc_col
        ]
        .max()
    )

    # ---------- Wide table by (sigma, sigma_blur) ---------- #
    wide = d_best.pivot_table(
        index=[args.sigma_col, args.sigma_blur_col],
        columns=args.dataset_col,
        values=args.dataset_auroc_col,
        aggfunc="max",
    ).reset_index()

    # Rename dataset columns to "<dataset> AUROC"
    wide.columns = [
        c if c in (args.sigma_col, args.sigma_blur_col) else f"{c} AUROC" for c in wide.columns
    ]

    # Merge global metrics (global_auroc and optional global_auroc_wo_sid)
    wide = wide.merge(g, on=[args.sigma_col, args.sigma_blur_col], how="left")

    # Column ordering: sigma, sigma_blur, global(s), then per-dataset AUROCs
    dataset_cols = sorted([c for c in wide.columns if c.endswith(" AUROC")])
    ordered_cols = [args.sigma_col, args.sigma_blur_col, args.global_auroc_col]
    if has_wo_sid:
        ordered_cols.append(args.maybe_global_auroc_wo_sid)
    ordered_cols += dataset_cols

    wide = wide[ordered_cols].sort_values(
        [args.sigma_col, args.sigma_blur_col]
    ).reset_index(drop=True)

    # Round all metrics (keep sigma/sigma_blur as floats without rounding)
    for c in wide.columns:
        if c not in (args.sigma_col, args.sigma_blur_col):
            wide[c] = wide[c].round(args.round)

    # Save the complete wide table
    table_by_sigma_csv = out_dir / "table_by_sigma.csv"
    wide.to_csv(table_by_sigma_csv, index=False)

    # ---------- Top-10 selection with tags ---------- #
    # Metrics to rank on: global, per-dataset AUROC, and maybe global_auroc_wo_sid
    metrics_to_rank = [args.global_auroc_col] + dataset_cols
    if has_wo_sid:
        metrics_to_rank.append(args.maybe_global_auroc_wo_sid)

    # Helper to convert column names into compact tags
    def metric_to_tag(m: str) -> str:
        if m == args.global_auroc_col:
            return "global"
        if has_wo_sid and m == args.maybe_global_auroc_wo_sid:
            return "global_wo_sid"
        if m.endswith(" AUROC"):
            return m.replace(" AUROC", "")
        return m

    # For each metric, take Top-10 rows and record which indices were selected
    tags_by_idx: dict[int, set[str]] = {}
    for m in metrics_to_rank:
        # Only consider rows where this metric is not NaN
        top_idx = wide[wide[m].notna()].nlargest(10, m).index
        tag = metric_to_tag(m)
        for i in top_idx:
            tags_by_idx.setdefault(i, set()).add(tag)

    # Build union of all Top-10 rows and attach a "top10_for" column
    top_union = wide.loc[sorted(tags_by_idx.keys())].copy()
    top_union.insert(
        2,  # after sigma and sigma_blur
        "top10_for",
        top_union.index.map(lambda i: "; ".join(sorted(tags_by_idx.get(i, [])))),
    )

    top_union_csv = out_dir / "top_union_table_by_sigma.csv"
    top_union.to_csv(top_union_csv, index=False)

    # ---------- Console summary ---------- #
    with pd.option_context("display.max_columns", None, "display.width", None):
        print("\n=== table_by_sigma (head) ===")
        print(wide.head())
        print("\n=== Top-union rows (head) ===")
        print(top_union.head())

    print("\n[OK] Wrote:")
    print(f" - {table_by_sigma_csv}")
    print(f" - {top_union_csv}  (Top-10 union with 'top10_for')")


if __name__ == "__main__":
    main()
