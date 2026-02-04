import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# =========================================
# CONSTANTS & CONFIGURATION
# =========================================
RAW_DATA_PATH = '../data/raw/manufacturing_dataset_1000_samples.csv'
OUTPUT_PATH = '../data/interim/cleaned_data.csv'
OUTPUT_DIR = os.path.dirname(OUTPUT_PATH)

TARGET_COL = 'Parts_Per_Hour'

CAT_COLS = [
    'Shift',
    'Machine_Type',
    'Material_Grade',
    'Day_of_Week'
]

NUM_COLS = [
    'Injection_Temperature',
    'Injection_Pressure',
    'Cycle_Time',
    'Cooling_Time',
    'Material_Viscosity',
    'Ambient_Temperature',
    'Machine_Age',
    'Operator_Experience',
    'Maintenance_Hours',
    'Machine_Utilization',
    'Temperature_Pressure_Ratio',
    'Total_Cycle_Time',
    'Efficiency_Score'
]

def run_pipeline():
    # ## 1. Load Data
    # Load dataset
    print("Loading data...")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Input file not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    initial_shape = df.shape
    print(f"Initial Data Shape: {initial_shape}")
    print("Columns:", df.columns.tolist())

    # ## 2. Column Management and Cleaning
    # Explicitly drop Timestamp if it exists
    if 'Timestamp' in df.columns:
        print("Dropping 'Timestamp' column...")
        df = df.drop(columns=['Timestamp'])
    else:
        print("'Timestamp' column not found, skipping drop.")

    # ## 3. Handle Missing Values
    print("Handling missing values...")
    # Identify actual numerical and categorical columns present in the dataframe
    # (Intersection with defined constants to be safe)
    present_num_cols = [c for c in NUM_COLS if c in df.columns]
    present_cat_cols = [c for c in CAT_COLS if c in df.columns]

    # Impute Numerical with Median
    for col in present_num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Impute Categorical with Mode
    for col in present_cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # Brief assertion to ensure no NaNs in feature columns
    assert df[present_num_cols + present_cat_cols].isnull().sum().sum() == 0, "Missing values remain after imputation!"

    # ## 4. Feature Encoding (One-Hot)
    print("Encoding categorical features...")
    # One-hot encode using pandas get_dummies
    df_encoded = pd.get_dummies(df, columns=present_cat_cols, dtype=int)

    print(f"Shape after encoding: {df_encoded.shape}")

    # ## 5. Feature Scaling
    print("Scaling numerical features...")
    # Initialize StandardScaler
    scaler = StandardScaler()

    # Apply scaling only to the numerical columns defined
    df_encoded[present_num_cols] = scaler.fit_transform(df_encoded[present_num_cols])

    # Check stats of a scaled column (mean should be approx 0, std approx 1)
    if present_num_cols:
        sample_col = present_num_cols[0]
        print(f"Stats for {sample_col} after scaling: Mean={df_encoded[sample_col].mean():.4f}, Std={df_encoded[sample_col].std():.4f}")

    # ## 6. Validation
    print("Validating processed data...")

    # 1. No missing values
    assert df_encoded.isnull().sum().sum() == 0, "Final dataset contains missing values!"

    # 2. Target column preserved
    assert TARGET_COL in df_encoded.columns, f"Target column '{TARGET_COL}' is missing!"

    # 3. Row count unchanged
    assert df_encoded.shape[0] == initial_shape[0], f"Row count changed! Initial: {initial_shape[0]}, Final: {df_encoded.shape[0]}"

    # 4. All features numeric
    # Check if all dtypes are numeric (int or float) - Exclude object type
    non_numeric_cols = df_encoded.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        print(f"Warning: Non-numeric columns found: {non_numeric_cols}")
    assert len(non_numeric_cols) == 0, "Final dataset contains non-numeric columns!"

    print("All validations passed.")

    # ## 7. Save Output
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Saving processed data to {OUTPUT_PATH}...")
    df_encoded.to_csv(OUTPUT_PATH, index=False)
    print("Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()
