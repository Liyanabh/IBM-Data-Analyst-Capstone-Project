import pandas as pd
import numpy as np
import re
import os
import json

# ---------- CONFIG ----------
FILE_PATH = r"C:\Users\liyana_bh\Downloads\DATA\LAB25_survey_data_Initial Data.csv"   # <- change to your local path if needed
OUTPUT_DIR = os.path.dirname(FILE_PATH) or "."
CLEAN_CSV = os.path.join(OUTPUT_DIR, "survey_data_cleaned.csv")
METRICS_JSON = os.path.join(OUTPUT_DIR, "wrangling_metrics.json")

COUNTRY_COL = "Country"
EMPLOYMENT_COL = "Employment"
RESP_ID_COL = "ResponseId"
COMP_COLS = ["ConvertedCompYearly", "CompTotal"]
YEARS_PRO_COL = "YearsCodePro"
JOBSAT_PREFIX = "JobSat"
TECH_MULTISELECT_COLS = [
    "LanguageHaveWorkedWith","LanguageWantToWorkWith",
    "DatabaseHaveWorkedWith","DatabaseWantToWorkWith",
    "PlatformHaveWorkedWith","PlatformWantToWorkWith",
    "WebFrameHaveWorkedWith","WebframeWantToWorkWith"
]

def to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == "O":
        cleaned = (series.astype(str)
                   .str.replace(",", "", regex=False)
                   .str.replace("$", "", regex=False)
                   .str.strip()
                   .replace({"None": np.nan, "nan": np.nan, "": np.nan}))
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")

def explode_multiselect(df, id_col, multi_col, sep=";"):
    if multi_col not in df.columns or id_col not in df.columns:
        return pd.DataFrame(columns=[id_col, multi_col])
    temp = df[[id_col, multi_col]].copy()
    temp[multi_col] = temp[multi_col].fillna("").astype(str).str.split(sep)
    temp = temp.explode(multi_col)
    temp[multi_col] = temp[multi_col].str.strip()
    temp = temp[temp[multi_col] != ""]
    return temp.reset_index(drop=True)

def standardize_label(x: str) -> str:
    if pd.isna(x) or str(x).strip() == "":
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("Javascript", "JavaScript")
    s = s.replace("NodeJs", "Node.js")
    s = s.replace("C Sharp", "C#")
    s = s.replace("C Plus Plus", "C++")
    s = s.replace("Postgres", "PostgreSQL")
    return s

# ---------- LOAD ----------
df = pd.read_csv(FILE_PATH, low_memory=False)

initial_rows, initial_cols = df.shape
countries_represented = df[COUNTRY_COL].nunique(dropna=True) if COUNTRY_COL in df.columns else 0
dup_count = df.duplicated(keep="first").sum()

# Clean copy
df_clean = df.drop_duplicates(keep="first").copy()

# Missingness (original columns only)
orig_cols = list(df.columns)
missing_before_orig = df[orig_cols].isna().sum().sum()
cells_before_orig = df[orig_cols].shape[0] * df[orig_cols].shape[1]

# Compensation numeric + outlier clipping (1–99 pct)
outlier_cases = 0
for c in COMP_COLS:
    if c in df_clean.columns:
        df_clean[c] = to_numeric(df_clean[c])
        col = df_clean[c]
        valid = col.dropna()
        if not valid.empty:
            lo, hi = np.percentile(valid, [1, 99])
            outlier_cases += int(((col < lo) | (col > hi)).sum())
            df_clean[c] = col.clip(lower=lo, upper=hi)

# Normalized comp columns
for c in COMP_COLS:
    if c in df_clean.columns:
        col = df_clean[c]
        if col.notna().sum() > 0 and col.min() != col.max():
            df_clean[c+"_norm"] = (col - col.min()) / (col.max() - col.min())
        else:
            df_clean[c+"_norm"] = np.nan

# Experience groups
if YEARS_PRO_COL in df_clean.columns:
    y = df_clean[YEARS_PRO_COL].astype(str)
    y = y.replace({"Less than 1 year": "0.5", "More than 50 years": "51"}, regex=False)
    y = y.str.replace(r"\D+", "", regex=True)
    y = pd.to_numeric(y, errors="coerce")
    df_clean[YEARS_PRO_COL+"_num"] = y
    bins = [-0.01, 1, 3, 5, 10, 20, 1000]
    labels = ["<1","1-3","3-5","5-10","10-20","20+"]
    df_clean["ExperienceGroup"] = pd.cut(y, bins=bins, labels=labels, include_lowest=True)

# Job satisfaction normalization (0–100)
jobsat_cols = [c for c in df_clean.columns if c.startswith(JOBSAT_PREFIX)]
jobsat_numeric_cols = []
for c in jobsat_cols:
    coln = to_numeric(df_clean[c])
    if coln.notna().sum() > 0:
        jobsat_numeric_cols.append(coln)
if jobsat_numeric_cols:
    jobsat_stack = pd.concat(jobsat_numeric_cols, axis=1)
    for c in jobsat_stack.columns:
        col = jobsat_stack[c]
        rng = col.max() - col.min()
        if pd.notna(rng) and rng != 0:
            jobsat_stack[c] = (col - col.min()) / rng * 100.0
    df_clean["JobSat_Score"] = jobsat_stack.mean(axis=1, skipna=True)

# Standardize & collect tech categories
TECH_MULTISELECT_COLS = [
    "LanguageHaveWorkedWith","LanguageWantToWorkWith",
    "DatabaseHaveWorkedWith","DatabaseWantToWorkWith",
    "PlatformHaveWorkedWith","PlatformWantToWorkWith",
    "WebFrameHaveWorkedWith","WebframeWantToWorkWith"
]
tech_unique = set()
for col in TECH_MULTISELECT_COLS:
    if col in df_clean.columns and RESP_ID_COL in df_clean.columns:
        temp = explode_multiselect(df_clean, RESP_ID_COL, col, sep=";")
        temp[col] = temp[col].map(standardize_label)
        temp = temp[temp[col] != ""]
        tech_unique.update(temp[col].unique().tolist())
tech_categories_standardized = len(tech_unique)

# Salary bands
salary_source = "ConvertedCompYearly" if "ConvertedCompYearly" in df_clean.columns else ("CompTotal" if "CompTotal" in df_clean.columns else None)
salary_bands_created = 0
if salary_source:
    s = df_clean[salary_source].dropna()
    if not s.empty:
        df_clean["SalaryBand"] = pd.qcut(s, q=7, duplicates="drop")
        salary_bands_created = df_clean["SalaryBand"].nunique(dropna=True)

# Missingness after (original columns only)
missing_after_orig = df_clean[orig_cols].isna().sum().sum()
cells_after_orig = df_clean[orig_cols].shape[0] * df_clean[orig_cols].shape[1]

accuracy_before_pct_orig = round((1 - (missing_before_orig / cells_before_orig)) * 100, 2) if cells_before_orig else None
accuracy_after_pct_orig  = round((1 - (missing_after_orig  / cells_after_orig )) * 100, 2) if cells_after_orig  else None
missing_after_pct_orig   = round((missing_after_orig / cells_after_orig) * 100, 2) if cells_after_orig else None

# Valid responses (non-null ResponseId)
valid_responses = int(df_clean[RESP_ID_COL].notna().sum()) if RESP_ID_COL in df_clean.columns else int(df_clean.shape[0])

# Demographic completeness (Age, Country, EdLevel)
demo_cols = [c for c in ["Age","Country","EdLevel"] if c in df_clean.columns]
demo_complete_rows = int(df_clean[demo_cols].notna().all(axis=1).sum()) if demo_cols else 0
demo_complete_pct = round(demo_complete_rows / df_clean.shape[0] * 100, 2) if df_clean.shape[0] else None

# Job titles categories
job_title_cats = int(df_clean[EMPLOYMENT_COL].astype(str).replace({"nan": np.nan}).nunique(dropna=True)) if EMPLOYMENT_COL in df_clean.columns else 0

# Technology stacks unique combos (HaveWorkedWith only)
have_cols = [c for c in TECH_MULTISELECT_COLS if ("HaveWorkedWith" in c and c in df_clean.columns)]
tech_stacks_unique = 0
if have_cols:
    stacks = df_clean[have_cols].fillna("").astype(str).agg(";".join, axis=1).str.replace(r";{2,}", ";", regex=True).str.strip("; ")
    tech_stacks_unique = int(stacks.nunique())

# Salary validity
validated_salary_pct = None
if salary_source:
    valid_salary = df_clean[salary_source].notna().sum()
    validated_salary_pct = round(valid_salary / df_clean.shape[0] * 100, 2)

# Save cleaned data
df_clean.to_csv(CLEAN_CSV, index=False)

metrics = {
    "initial": {
        "raw_records": int(df.shape[0]),
        "countries_represented": int(df[COUNTRY_COL].nunique(dropna=True)) if COUNTRY_COL in df.columns else 0,
        "questions_analyzed": int(df.shape[1]),
        "duplicate_rows": int(dup_count),
    },
    "after_cleaning": {
        "clean_records": int(df_clean.shape[0]),
        "removed_duplicates": int(dup_count),
        "handled_missing_values_cells": 0,  # unchanged because no imputation
        "valid_responses": int(valid_responses),
        "valid_responses_pct_of_original": round(valid_responses / df.shape[0] * 100, 2) if df.shape[0] else None,
        "countries_represented": int(df_clean[COUNTRY_COL].nunique(dropna=True)) if COUNTRY_COL in df_clean.columns else 0,
    },
    "data_quality": {
        "accuracy_before_pct": accuracy_before_pct_orig,
        "accuracy_after_pct": accuracy_after_pct_orig,
        "missing_data_after_pct": missing_after_pct_orig,
        "outliers_identified_treated": int(outlier_cases),
        "response_consistency_improvement_pct": 0.0
    },
    "final_features": {
        "clean_demographic_completeness_pct": demo_complete_pct,
        "standardized_job_titles_categories": job_title_cats,
        "technology_stacks_unique_combinations": tech_stacks_unique,
        "validated_salary_accuracy_pct": validated_salary_pct
    },
    "key_metrics_processed": {
        "jobsat_score_normalized": bool("JobSat_Score" in df_clean.columns),
        "tech_categories_standardized": int(tech_categories_standardized),
        "salary_bands_created": int(salary_bands_created),
        "experience_groups_defined": int(df_clean["ExperienceGroup"].nunique(dropna=True)) if "ExperienceGroup" in df_clean.columns else 0,
    }
}

with open(METRICS_JSON, "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
print(f"\nSaved cleaned CSV to: {CLEAN_CSV}")
print(f"Saved metrics JSON to: {METRICS_JSON}")
