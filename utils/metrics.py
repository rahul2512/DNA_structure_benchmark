"""
utils/metrics.py
================
Core data-loading, validation, classification, and statistical functions
for the Tetramer Structural Comparison application.

All functions are pure (no Shiny/UI dependencies) so they can be tested
independently of the application layer.
"""

from __future__ import annotations

import io
import re
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_BASES = set("ACGT")
INFORMATION_TYPES = ("Shape", "Variance")
PURINES = set("AG")
PYRIMIDINES = set("CT")

STEP_CLASS_LABELS = {
    "RR": "RR — Purine/Purine",
    "RY": "RY — Purine/Pyrimidine",
    "YR": "YR — Pyrimidine/Purine",
    "YY": "YY — Pyrimidine/Pyrimidine",
}

# Colourblind-friendly palette (Wong 2011)
CLASS_COLOURS = {
    "RR": "#E69F00",   # orange
    "RY": "#56B4E9",   # sky blue
    "YR": "#009E73",   # teal
    "YY": "#CC79A7",   # reddish purple
}

# ---------------------------------------------------------------------------
# Dataset loading & validation
# ---------------------------------------------------------------------------

def load_dataset(file_or_path) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load a CSV from a file-like object, path string, or bytes.

    Returns
    -------
    (df, error_message)
        df is None when loading fails; error_message is None on success.
    """
    try:
        if isinstance(file_or_path, (str, bytes, bytearray)):
            if isinstance(file_or_path, str):
                df = pd.read_csv(file_or_path)
            else:
                df = pd.read_csv(io.BytesIO(file_or_path))
        else:
            # file-like object or Shiny UploadFile
            if hasattr(file_or_path, "read"):
                content = file_or_path.read()
                df = pd.read_csv(io.BytesIO(content) if isinstance(content, bytes) else io.StringIO(content))
            else:
                df = pd.read_csv(file_or_path)
    except Exception as exc:
        return None, f"Could not read file: {exc}"

    return df, None


def validate_dataset(df: pd.DataFrame, name: str = "Dataset") -> tuple[Optional[pd.DataFrame], list[str]]:
    """
    Validate a loaded DataFrame.

    Checks
    ------
    * First column is 'Tetramer'
    * Optional 'Info' column: 'Shape' (mean structural parameters) or 'Variance'.
      If absent, every row is treated as Shape.
    * Every sequence is a 4-base (tetramer) or 2-base (dimer) DNA string using A/C/G/T;
      tetramer and dimer rows may be mixed in the same file
    * No duplicate (sequence, Info) pairs
    * Remaining columns are numeric (non-numeric columns are dropped with a warning)
    * Warns if there are not 256 tetramers / 16 dimers (when that level is present)

    Returns
    -------
    (cleaned_df, warnings_list)
        cleaned_df is None if a fatal error is found.
    """
    warnings: list[str] = []

    if df is None or df.empty:
        return None, [f"{name}: empty or unreadable file."]

    # ---- Tetramer column ----
    cols = list(df.columns)
    if cols[0].strip() != "Tetramer":
        # Try to find it anywhere
        lower_cols = [c.strip() for c in cols]
        if "Tetramer" in lower_cols:
            idx = lower_cols.index("Tetramer")
            # Move it to front
            new_order = [cols[idx]] + [c for i, c in enumerate(cols) if i != idx]
            df = df[new_order]
            df.columns = ["Tetramer"] + list(df.columns[1:])
            warnings.append(f"{name}: 'Tetramer' column moved to first position.")
        else:
            return None, [f"{name}: first column must be 'Tetramer'. Got '{cols[0]}'."]

    df = df.copy()
    df["Tetramer"] = df["Tetramer"].astype(str).str.strip().str.upper()

    # ---- Validate tetramer strings ----
    invalid_mask = ~df["Tetramer"].apply(_is_valid_tetramer)
    if invalid_mask.any():
        bad = df.loc[invalid_mask, "Tetramer"].tolist()[:5]
        return None, [
            f"{name}: invalid sequences found (e.g. {bad}). "
            "Each entry must be 4 (tetramer) or 2 (dimer) characters from A/C/G/T."
        ]

    # ---- Info column (Shape / Variance) ----
    info_col = next((c for c in df.columns if str(c).strip().lower() in ("info", "information")), None)
    if info_col is None:
        df["Info"] = "Shape"
    else:
        if info_col != "Info":
            df = df.rename(columns={info_col: "Info"})
        df["Info"] = df["Info"].astype(str).str.strip().str.capitalize()
        bad_info = sorted(set(df["Info"]) - set(INFORMATION_TYPES))
        if bad_info:
            return None, [
                f"{name}: invalid Info values {bad_info}. "
                f"Allowed values: {', '.join(INFORMATION_TYPES)}."
            ]
        order = ["Tetramer", "Info"] + [c for c in df.columns if c not in ("Tetramer", "Info")]
        df = df[order]

    # ---- Duplicates ----
    dupes = df.duplicated(subset=["Tetramer", "Info"])
    if dupes.any():
        dup_list = df.loc[dupes, ["Tetramer", "Info"]].astype(str).agg("/".join, axis=1).tolist()[:5]
        return None, [f"{name}: duplicate (sequence, Info) rows found: {dup_list}."]

    # ---- Row count warnings (per Info type) ----
    seq_len = df["Tetramer"].str.len()
    for info in INFORMATION_TYPES:
        sel = df["Info"] == info
        n_tet, n_dim = int((sel & (seq_len == 4)).sum()), int((sel & (seq_len == 2)).sum())
        if n_tet and n_tet != 256:
            warnings.append(f"{name}: expected 256 {info} tetramers, found {n_tet}.")
        if n_dim and n_dim != 16:
            warnings.append(f"{name}: expected 16 {info} dimers, found {n_dim}.")

    # ---- Numeric coordinate columns ----
    coord_cols = [c for c in df.columns if c not in ("Tetramer", "Info")]
    non_numeric = []
    for c in coord_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                n_nan = df[c].isna().sum()
                if n_nan > 0:
                    warnings.append(f"{name}: column '{c}' has {n_nan} non-numeric values (set to NaN).")
            except Exception:
                non_numeric.append(c)

    if non_numeric:
        warnings.append(f"{name}: dropping non-numeric columns: {non_numeric}.")
        df = df.drop(columns=non_numeric)

    coord_cols_clean = [c for c in df.columns if c not in ("Tetramer", "Info")]
    if not coord_cols_clean:
        return None, [f"{name}: no numeric coordinate columns found after cleaning."]

    # Set Tetramer as index for easy merging later
    df = df.set_index("Tetramer")

    return df, warnings


def _is_valid_tetramer(s: str) -> bool:
    """Valid sequence = tetramer (4 nt) or dimer (2 nt) over A/C/G/T."""
    return len(s) in (2, 4) and all(c in VALID_BASES for c in s)


def select_view(df: Optional[pd.DataFrame], level: str, info: str) -> Optional[pd.DataFrame]:
    """
    Return the numeric table for one (level, information) combination, indexed by sequence.

    level : "tet" (4-letter rows) or "dim" (2-letter rows)
    info  : "Shape" or "Variance" (value of the Info column)
    Returns None when the dataset has no such rows.
    """
    if df is None or "Info" not in df.columns:
        return None
    want_len = 2 if level == "dim" else 4
    mask = (df.index.str.len() == want_len) & (df["Info"] == info)
    sub = df.loc[mask].drop(columns="Info")
    return sub if len(sub) else None


def describe_dataset(df: Optional[pd.DataFrame]) -> dict:
    """Row counts per (Info, level), e.g. {"Shape": {"tet": 256, "dim": 16}, ...}."""
    out: dict = {}
    if df is None or "Info" not in df.columns:
        return out
    n = df.index.str.len()
    for info in INFORMATION_TYPES:
        sel = df["Info"] == info
        nt, nd = int((sel & (n == 4)).sum()), int((sel & (n == 2)).sum())
        if nt or nd:
            out[info] = {"tet": nt, "dim": nd}
    return out


def check_dataset_compatibility(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_a: str = "Dataset A",
    name_b: str = "Dataset B",
) -> tuple[list[str], list[str], list[str]]:
    """
    Compare coordinate column sets of two validated DataFrames (Tetramer as index).

    Returns
    -------
    (shared_coords, only_in_a, only_in_b)
    """
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    shared = sorted(cols_a & cols_b)
    only_a = sorted(cols_a - cols_b)
    only_b = sorted(cols_b - cols_a)
    return shared, only_a, only_b


# ---------------------------------------------------------------------------
# Central dinucleotide classification
# ---------------------------------------------------------------------------

def classify_central_step(tetramer: str) -> tuple[str, str]:
    """
    Return (central_dinucleotide, step_class) for a tetramer or dimer string.

    Tetramer: central dinucleotide = positions 2 and 3 (1-indexed), i.e. tetramer[1:3].
    Dimer (2-letter entry): the dimer itself is the dinucleotide.

    Step classes
    ------------
    RR  purine–purine
    RY  purine–pyrimidine
    YR  pyrimidine–purine
    YY  pyrimidine–pyrimidine
    """
    tetramer = tetramer.upper()
    central = tetramer if len(tetramer) == 2 else tetramer[1:3]
    b1, b2 = central[0], central[1]
    c1 = "R" if b1 in PURINES else "Y"
    c2 = "R" if b2 in PURINES else "Y"
    return central, c1 + c2


def add_classification_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'Central_step' and 'Step_class' columns to a long-form DataFrame
    that has a 'Tetramer' column (or Tetramer as index).

    Works whether Tetramer is a column or the index.
    """
    df = df.copy()
    if "Tetramer" in df.columns:
        tetramers = df["Tetramer"]
    else:
        tetramers = df.index.to_series()

    df["Central_step"] = tetramers.apply(lambda t: classify_central_step(t)[0])
    df["Step_class"]   = tetramers.apply(lambda t: classify_central_step(t)[1])
    return df


# ---------------------------------------------------------------------------
# Dataset preparation / merging
# ---------------------------------------------------------------------------

def prepare_comparison(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    coord: str,
    name_x: str = "X",
    name_y: str = "Y",
) -> pd.DataFrame:
    """
    Merge two validated DataFrames (Tetramer as index) on the shared tetramer set
    for a single coordinate column.

    Returns a flat DataFrame with columns:
        Tetramer, Central_step, Step_class, X_val, Y_val, Difference, Abs_difference
    """
    # Only shared tetramers
    shared_idx = df_x.index.intersection(df_y.index)

    x_vals = df_x.loc[shared_idx, coord]
    y_vals = df_y.loc[shared_idx, coord]

    cmp = pd.DataFrame({
        "Tetramer":    shared_idx,
        f"{name_x}":   x_vals.values,
        f"{name_y}":   y_vals.values,
    })

    # Remove rows where either value is NaN
    cmp = cmp.dropna(subset=[name_x, name_y])

    cmp = add_classification_columns(cmp)

    cmp["Difference"]     = cmp[name_y] - cmp[name_x]
    cmp["Abs_difference"] = cmp["Difference"].abs()

    return cmp.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistical metrics
# ---------------------------------------------------------------------------

def calculate_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Pearson r and two-tailed p-value.
    Returns (nan, nan) for degenerate inputs.
    """
    x, y = _clean_pair(x, y)
    if len(x) < 3:
        return float("nan"), float("nan")
    try:
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return float("nan"), float("nan")


def calculate_cosine(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cosine similarity = dot(x, y) / (||x|| * ||y||).
    Returns nan for zero-norm vectors.
    """
    x, y = _clean_pair(x, y)
    if len(x) < 1:
        return float("nan")
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return float("nan")
    return float(np.dot(x, y) / (norm_x * norm_y))


def calculate_rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Root mean square error (y - x)."""
    x, y = _clean_pair(x, y)
    if len(x) < 1:
        return float("nan")
    return float(np.sqrt(np.mean((y - x) ** 2)))


def calculate_summary_statistics(
    x: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """
    Calculate a dictionary of summary statistics for the difference (y - x).
    """
    x, y = _clean_pair(x, y)
    if len(x) < 1:
        return {k: float("nan") for k in
                ("n", "mean_abs_diff", "median_abs_diff", "rmse", "max_abs_diff", "pearson_r", "cosine")}
    diff = y - x
    abs_diff = np.abs(diff)
    r, _ = calculate_pearson(x, y)
    cos   = calculate_cosine(x, y)
    return {
        "n":                len(x),
        "mean_abs_diff":    float(np.mean(abs_diff)),
        "median_abs_diff":  float(np.median(abs_diff)),
        "rmse":             float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_diff":     float(np.max(abs_diff)),
        "pearson_r":        r,
        "cosine":           cos,
    }


def calculate_all_coords_stats(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    coords: list[str],
) -> pd.DataFrame:
    """
    For every coordinate in `coords`, calculate Pearson r and cosine similarity
    using only the shared tetramer set.

    Returns a DataFrame with columns: Coordinate, Pearson_r, Cosine, N
    """
    shared_idx = df_x.index.intersection(df_y.index)
    rows = []
    for c in coords:
        if c not in df_x.columns or c not in df_y.columns:
            rows.append({"Coordinate": c, "Pearson_r": np.nan, "Cosine": np.nan, "N": 0})
            continue
        x = df_x.loc[shared_idx, c].values.astype(float)
        y = df_y.loc[shared_idx, c].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        r, _ = calculate_pearson(x, y)
        cos   = calculate_cosine(x, y)
        rows.append({"Coordinate": c, "Pearson_r": r, "Cosine": cos, "N": len(x)})
    return pd.DataFrame(rows)


def calculate_heatmap_stats(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    coords: list[str],
    metric: str = "pearson",
) -> pd.DataFrame:
    """
    For each coordinate × step-class combination, calculate Pearson r or cosine.

    Returns a DataFrame with coords as rows and ['RR','RY','YR','YY','All'] as cols.
    """
    shared_idx = df_x.index.intersection(df_y.index)

    # Build a merged frame with classification
    merged = pd.DataFrame(index=shared_idx)
    merged.index.name = "Tetramer"
    merged["Central_step"] = merged.index.to_series().apply(lambda t: classify_central_step(t)[0])
    merged["Step_class"]   = merged.index.to_series().apply(lambda t: classify_central_step(t)[1])

    classes = ["RR", "RY", "YR", "YY", "All"]
    result = pd.DataFrame(index=coords, columns=classes, dtype=float)

    for c in coords:
        if c not in df_x.columns or c not in df_y.columns:
            result.loc[c] = np.nan
            continue
        x_all = df_x.loc[shared_idx, c].values.astype(float)
        y_all = df_y.loc[shared_idx, c].values.astype(float)
        sc_all = merged["Step_class"].values

        for cls in classes:
            if cls == "All":
                mask = ~(np.isnan(x_all) | np.isnan(y_all))
            else:
                mask = (sc_all == cls) & ~(np.isnan(x_all) | np.isnan(y_all))
            xm, ym = x_all[mask], y_all[mask]
            if metric == "pearson":
                val, _ = calculate_pearson(xm, ym)
            else:
                val = calculate_cosine(xm, ym)
            result.loc[c, cls] = val

    return result.astype(float)


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def linear_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Fit y = a*x + b via least squares.
    Returns dict with keys: slope, intercept, r_squared, x_line, y_line
    """
    x, y = _clean_pair(x, y)
    if len(x) < 2:
        return {}
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept
    return {
        "slope":     float(slope),
        "intercept": float(intercept),
        "r_squared": float(r2),
        "x_line":    x_line,
        "y_line":    y_line,
    }


# ---------------------------------------------------------------------------
# Coordinate type detection
# ---------------------------------------------------------------------------

def detect_coord_type(name: str) -> str:
    """
    Return 'Intra', 'Inter', or 'Unknown' based on coordinate name heuristics.
    """
    n = name.lower()
    if "intra" in n:
        return "Intra"
    if "inter" in n:
        return "Inter"
    # Common intra-bp parameters
    intra_keywords = {"buckle", "propeller", "opening", "shear", "stretch", "stagger"}
    # Common inter-bp parameters
    inter_keywords = {"shift", "slide", "rise", "tilt", "roll", "twist"}
    for kw in intra_keywords:
        if kw in n:
            return "Intra"
    for kw in inter_keywords:
        if kw in n:
            return "Inter"
    return "Unknown"


def get_coord_type_label(name: str) -> str:
    t = detect_coord_type(name)
    if t == "Unknown":
        return ""
    return f"({t})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Convert to float arrays and remove rows where either is NaN."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    # Trim to same length (should already be aligned but belt-and-suspenders)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    mask = ~(np.isnan(x) | np.isnan(y))
    return x[mask], y[mask]
