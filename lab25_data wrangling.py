# wrangle_for_dashboards.py
# Python 3.9+
# ---------------------------------------------------------
# STEP 1: Identify & remove duplicate rows
# STEP 2: Analyze missing values
# STEP 3: Examine Employment distribution
# STEP 4: Normalize CompTotal & ConvertedCompYearly
# STEP 5: Save cleaned data as survey_data_cleaned.csv
#
# Bonus (for dashboards): produce exploded (tidy) tables for
#   multi-select columns like LanguageHaveWorkedWith, etc.
# ---------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== EDIT THIS: point to YOUR CSV file ==========
# Example (Windows):
FILE_PATH = r"C:\Users\liyana_bh\Downloads\survey_data_updated 5 (1).csv"
# =======================================================

# Output locations (saved next to your CSV by default)
BASE_DIR = os.path.dirname(FILE_PATH)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
CLEAN_CSV = os.path.join(BASE_DIR, "survey_data_cleaned.csv")
MISSING_SUMMARY_CSV = os.path.join(BASE_DIR, "missing_values_summary.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Helpful helpers
def to_numeric(series: pd.Series) -> pd.Series:
    """Safely coerce to numeric (removes commas, $, blanks -> NaN)."""
    if series.dtype == "O":
        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
            .replace({"None": np.nan, "nan": np.nan, "": np.nan})
        )
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")

def explode_multiselect(df: pd.DataFrame, id_col: str, multi_col: str, sep=";") -> pd.DataFrame:
    """
    Takes a dataset where 'multi_col' contains delimited values (e.g., 'Python; SQL')
    and returns a tidy table [id_col, multi_col_single] with one value per row.
    """
    if multi_col not in df.columns or id_col not in df.columns:
        return pd.DataFrame(columns=[id_col, multi_col])

    temp = df[[id_col, multi_col]].copy()
    temp[multi_col] = temp[multi_col].fillna("").astype(str)
    temp[multi_col] = temp[multi_col].str.split(sep)
    temp = temp.explode(multi_col)
    temp[multi_col] = temp[multi_col].str.strip()
    temp = temp[temp[multi_col] != ""]
    return temp.reset_index(drop=True)

print("Loading dataset...")
df = pd.read_csv(FILE_PATH, low_memory=False)

print("\n=== INITIAL DATASET ===")
print(f"Rows: {df.shape[0]:,} | Cols: {df.shape[1]}")
print("Columns:", list(df.columns))

# ---------------------------------------------------------
# Initial visuals (quick sanity checks)
# ---------------------------------------------------------

# 1) Missing values overview (initial)
missing_initial = df.isna().sum()
nz_missing_initial = missing_initial[missing_initial > 0].sort_values(ascending=False)

if len(nz_missing_initial) > 0:
    plt.figure(figsize=(12, 6))
    nz_missing_initial.plot(kind="bar")
    plt.title("Missing Values per Column (Initial Dataset)")
    plt.xlabel("Column")
    plt.ylabel("Count Missing")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "initial_missing_values.png"), dpi=150)
    plt.close()

# 2) Employment distribution (initial), if present
if "Employment" in df.columns:
    emp_counts_init = df["Employment"].value_counts(dropna=False).head(20)
    plt.figure(figsize=(10, 6))
    emp_counts_init.plot(kind="bar")
    plt.title("Employment Distribution (Initial)")
    plt.xlabel("Employment")
    plt.ylabel("Respondent Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "initial_employment_distribution.png"), dpi=150)
    plt.close()

# 3) Quick look at your dashboard key columns (top counts)
dashboard_cols = [
    "ResponseId","Age","EdLevel","Country",
    "LanguageHaveWorkedWith","LanguageWantToWorkWith",
    "DatabaseHaveWorkedWith","DatabaseWantToWorkWith",
    "PlatformHaveWorkedWith","PlatformWantToWorkWith",
    "WebFrameHaveWorkedWith","WebframeWantToWorkWith"
]
for col in dashboard_cols:
    if col in df.columns and df[col].dtype == "O":
        vc = df[col].value_counts(dropna=False).head(15)
        if len(vc) > 0:
            plt.figure(figsize=(10, 5))
            vc.plot(kind="bar")
            plt.title(f"Top Values - {col} (Initial)")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"initial_top_{col}.png"), dpi=150)
            plt.close()

# ---------------------------------------------------------
# STEP 1: Identify & remove duplicate rows
# ---------------------------------------------------------
print("\n=== STEP 1: Duplicates ===")
dups = df.duplicated(keep="first")
n_dups = dups.sum()
print(f"Duplicate rows found: {n_dups:,}")

df_clean = df.drop_duplicates(keep="first").copy()
print(f"After removing duplicates: {df_clean.shape[0]:,} rows remain")

# ---------------------------------------------------------
# STEP 2: Analyze missing values
# ---------------------------------------------------------
print("\n=== STEP 2: Missing Values (after duplicate removal) ===")
missing_counts = df_clean.isna().sum()
missing_pct = (missing_counts / len(df_clean) * 100).round(2)
missing_summary = pd.DataFrame({
    "missing_count": missing_counts.sort_values(ascending=False),
    "missing_pct": missing_pct.sort_values(ascending=False)
}).loc[missing_counts.sort_values(ascending=False).index]

print(missing_summary.head(25))  # print top 25 for quick view
missing_summary.to_csv(MISSING_SUMMARY_CSV)

# Plot missing (after duplicate removal)
nz_missing_clean = missing_counts[missing_counts > 0].sort_values(ascending=False)
if len(nz_missing_clean) > 0:
    plt.figure(figsize=(12, 6))
    nz_missing_clean.plot(kind="bar")
    plt.title("Missing Values per Column (Cleaned)")
    plt.xlabel("Column")
    plt.ylabel("Count Missing")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "missing_values_cleaned.png"), dpi=150)
    plt.close()

# ---------------------------------------------------------
# STEP 3: Examine employment status
# ---------------------------------------------------------
print("\n=== STEP 3: Employment Distribution (Cleaned) ===")
if "Employment" in df_clean.columns:
    emp_counts = df_clean["Employment"].value_counts(dropna=False)
    emp_pct = (emp_counts / len(df_clean) * 100).round(2)
    emp_summary = pd.DataFrame({"count": emp_counts, "pct": emp_pct})
    print(emp_summary.head(20))

    plt.figure(figsize=(10, 6))
    emp_counts.head(20).plot(kind="bar")
    plt.title("Employment Distribution (Top 20) - Cleaned")
    plt.xlabel("Employment")
    plt.ylabel("Respondent Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "employment_distribution_cleaned.png"), dpi=150)
    plt.close()
else:
    print("Column 'Employment' not found — skipping STEP 3 plot.")

# ---------------------------------------------------------
# STEP 4: Normalize CompTotal & ConvertedCompYearly
# ---------------------------------------------------------
print("\n=== STEP 4: Normalization ===")
norm_cols = []
for col in ["CompTotal", "ConvertedCompYearly"]:
    if col in df_clean.columns:
        df_clean[col] = to_numeric(df_clean[col])
        norm_cols.append(col)

if len(norm_cols) == 0:
    print("No 'CompTotal' or 'ConvertedCompYearly' found — skipping normalization.")
else:
    for col in norm_cols:
        min_v = df_clean[col].min(skipna=True)
        max_v = df_clean[col].max(skipna=True)
        out_col = f"{col}_norm"
        if pd.isna(min_v) or pd.isna(max_v) or min_v == max_v:
            df_clean[out_col] = np.nan
            print(f"- {col}: cannot normalize (all NaN/constant).")
        else:
            df_clean[out_col] = (df_clean[col] - min_v) / (max_v - min_v)
            print(f"- {col}: normalized to '{out_col}'  (min={min_v}, max={max_v})")

    # Visualize distributions and normalized comparison
    for col in norm_cols:
        valid = df_clean[col].dropna()
        if len(valid) > 0:
            plt.figure(figsize=(10, 5))
            plt.hist(valid, bins=40)
            plt.title(f"{col} Distribution (Cleaned)")
            plt.xlabel(col); plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{col}_hist_cleaned.png"), dpi=150)
            plt.close()

    if all(c in df_clean.columns for c in ["CompTotal_norm","ConvertedCompYearly_norm"]):
        both = df_clean[["CompTotal_norm","ConvertedCompYearly_norm"]].dropna()
        if len(both) > 0:
            plt.figure(figsize=(6, 6))
            plt.scatter(both["CompTotal_norm"], both["ConvertedCompYearly_norm"], s=8, alpha=0.5)
            plt.title("Normalized: CompTotal vs ConvertedCompYearly")
            plt.xlabel("CompTotal_norm"); plt.ylabel("ConvertedCompYearly_norm")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "normalized_comp_scatter.png"), dpi=150)
            plt.close()

# ---------------------------------------------------------
# STEP 5: Save the cleaned dataset
# ---------------------------------------------------------
df_clean.to_csv(CLEAN_CSV, index=False)
print(f"\nSaved cleaned dataset → {CLEAN_CSV}")
print(f"Missing-values summary → {MISSING_SUMMARY_CSV}")
print(f"Plots saved in → {PLOTS_DIR}")

# ---------------------------------------------------------
# BONUS: Build tidy (exploded) tables for dashboard fields
#       Saves CSVs like language_have_worked_exploded.csv, etc.
#       Each contains [ResponseId, <value>] one-per-row.
# ---------------------------------------------------------
print("\n=== BONUS: Exploded tables for dashboards ===")
id_col = "ResponseId" if "ResponseId" in df_clean.columns else None

multi_select_map = {
    "language_have_worked": "LanguageHaveWorkedWith",
    "language_want_work": "LanguageWantToWorkWith",
    "db_have_worked": "DatabaseHaveWorkedWith",
    "db_want_work": "DatabaseWantToWorkWith",
    "platform_have_worked": "PlatformHaveWorkedWith",
    "platform_want_work": "PlatformWantToWorkWith",
    "webframe_have_worked": "WebFrameHaveWorkedWith",
    "webframe_want_work": "WebframeWantToWorkWith",
}

if id_col:
    for tag, col in multi_select_map.items():
        if col in df_clean.columns:
            expl = explode_multiselect(df_clean, id_col=id_col, multi_col=col, sep=";")
            out_path = os.path.join(BASE_DIR, f"{tag}_exploded.csv")
            expl.to_csv(out_path, index=False)
            print(f"- Saved exploded table for {col} → {out_path}")
else:
    print("No 'ResponseId' column found; skipped exploded tables.")
