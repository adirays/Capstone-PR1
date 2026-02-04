# Data Preprocessing Notes

## Overview
This document details the preprocessing strategy implemented in `notebooks/01_data_preprocessing.ipynb` for the manufacturing dataset.

## Dataset Structure
- **Raw Data Path**: `data/raw/manufacturing_dataset_1000_samples.csv`
- **Output Data Path**: `data/interim/cleaned_data.csv`
- **Target Variable**: `Parts_Per_Hour`

## Preprocessing Strategy

### 1. Column Management
- **Dropped Columns**: 
  - `Timestamp`: Removed as it is not needed for the current modeling task and to prevent temporal leakage if not explicitly treated as a time-series feature. Logic ensures it is only dropped if present.
- **Preserved Columns**: All other columns from the raw dataset, including the target `Parts_Per_Hour`.

### 2. Missing Value Imputation
- **Numerical Features**: Imputed using the **median** to be robust against outliers.
- **Categorical Features**: Imputed using the **mode** (most frequent value) to preserve category distribution.

### 3. Feature Encoding
- **Categorical Features**: `Shift`, `Machine_Type`, `Material_Grade`, `Day_of_Week`
- **Method**: One-Hot Encoding (`pd.get_dummies`)
- **Reasoning**: To convert categorical variables into a machine-readable numeric format while maintaining interpretability. `drop_first=False` was used to retain all categories for explicit feature importance analysis.

### 4. Feature Scaling
- **Numerical Features**: `Injection_Temperature`, `Injection_Pressure`, `Cycle_Time`, `Cooling_Time`, `Material_Viscosity`, `Ambient_Temperature`, `Machine_Age`, `Operator_Experience`, `Maintenance_Hours`, `Machine_Utilization`, `Temperature_Pressure_Ratio`, `Total_Cycle_Time`, `Efficiency_Score`
- **Method**: `StandardScaler` (z-score normalization).
- **Reasoning**: To ensure all numerical features contribute equally to the model and to improve the convergence of gradient-based algorithms. feature centering at mean 0 and variance 1.

### 5. Final Feature Count
- The final dataset contains the original numerical columns (scaled), the one-hot encoded categorical columns, and the target column.
- **Verification**: The code asserts 'All features numeric' and 'No missing values'.

## Validation Steps
The pipeline includes strict assertions to ensure:
- Input shape is loaded correctly.
- No `NaN` values remain after processing.
- Target column `Parts_Per_Hour` is present and unchanged (except for row alignment if any drops occurred, though none are expected).
- Row count remains consistent throughout the pipeline.
- Output file is successfully written.
