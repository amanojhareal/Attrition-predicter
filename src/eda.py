"""
eda.py

Exploratory Data Analysis for the Employee Attrition dataset.
Plain script version — no Jupyter needed. Run it from the terminal:

    python src/eda.py

All charts are saved as PNG files inside eda_output/ so you can open
and review them directly, and all key numbers are printed to the console.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

RAW_PATH = "../data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_DIR = "eda_output"


def load_data():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded dataset — shape: {df.shape}")
    return df


def basic_structure(df):
    print("\n--- df.info() ---")
    df.info()

    print("\n--- df.describe() ---")
    print(df.describe())

    print("\n--- Missing values ---")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")


def check_constant_columns(df):
    print("\n--- Candidate columns to drop (low/no variance) ---")
    candidates = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
    for col in candidates:
        if col in df.columns:
            print(f"  {col}: {df[col].nunique()} unique values")


def class_balance(df, out_dir):
    print("\n--- Attrition class balance ---")
    counts = df["Attrition"].value_counts()
    pct = df["Attrition"].value_counts(normalize=True) * 100
    print(counts)
    print(pct.round(2))

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Attrition")
    plt.title("Attrition Class Balance")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/01_class_balance.png", dpi=150)
    plt.close()
    print(f"Saved -> {out_dir}/01_class_balance.png")


def correlation_heatmap(df, out_dir):
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(14, 10))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap - Numeric Features")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/02_correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved -> {out_dir}/02_correlation_heatmap.png")

    df_temp = df.copy()
    df_temp["Attrition_binary"] = df_temp["Attrition"].map({"Yes": 1, "No": 0})
    corr_with_target = (
        numeric_df.assign(Attrition_binary=df_temp["Attrition_binary"])
        .corr()["Attrition_binary"]
        .sort_values(ascending=False)
    )
    print("\n--- Correlation with Attrition (numeric features) ---")
    print(corr_with_target)


def categorical_vs_attrition(df, col, out_dir, index):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue="Attrition")
    plt.title(f"Attrition by {col}")
    plt.tight_layout()
    fname = f"{out_dir}/{index:02d}_{col}_vs_attrition.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved -> {fname}")

    crosstab = pd.crosstab(df[col], df["Attrition"], normalize="index") * 100
    print(f"\n--- {col} vs Attrition (%) ---")
    print(crosstab.round(1))


def numeric_vs_attrition(df, col, out_dir, index):
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=df, x="Attrition", y=col)
    plt.title(f"{col} vs Attrition")
    plt.tight_layout()
    fname = f"{out_dir}/{index:02d}_{col}_vs_attrition.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved -> {fname}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    basic_structure(df)
    check_constant_columns(df)
    class_balance(df, OUTPUT_DIR)
    correlation_heatmap(df, OUTPUT_DIR)

    # Categorical comparisons
    categorical_vs_attrition(df, "OverTime", OUTPUT_DIR, 3)
    categorical_vs_attrition(df, "JobSatisfaction", OUTPUT_DIR, 4)
    categorical_vs_attrition(df, "WorkLifeBalance", OUTPUT_DIR, 5)

    # Numeric comparisons
    numeric_vs_attrition(df, "YearsSinceLastPromotion", OUTPUT_DIR, 6)
    numeric_vs_attrition(df, "MonthlyIncome", OUTPUT_DIR, 7)
    numeric_vs_attrition(df, "Age", OUTPUT_DIR, 8)
    numeric_vs_attrition(df, "YearsAtCompany", OUTPUT_DIR, 9)

    print(f"\nAll charts saved in ./{OUTPUT_DIR}/ — open them directly to review.")
    print("""
--- Summary checklist (fill in after reviewing charts/numbers above) ---
- Attrition rate: ~___% -> confirms class imbalance -> use F1/ROC-AUC, not accuracy
- OverTime=Yes association with attrition: stronger / weaker than expected?
- Low JobSatisfaction (1-2) association with attrition: confirmed?
- Low WorkLifeBalance association with attrition: confirmed?
- MonthlyIncome pattern: lower income -> higher attrition?
- YearsAtCompany pattern: shorter tenure -> higher attrition?
""")


if __name__ == "__main__":
    main()