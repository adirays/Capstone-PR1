import pandas as pd
import numpy as np
import os

# Configuration
DATA_DIR = '../data/raw'
FILE_NAME = 'manufacturing_dataset_1000_samples.csv'
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)
NUM_SAMPLES = 1000

def generate_data():
    np.random.seed(42)
    
    # Categorical Columns
    shifts = ['Morning', 'Evening', 'Night']
    machine_types = ['Type_A', 'Type_B', 'Type_C']
    material_grades = ['Grade_1', 'Grade_2', 'Grade_3']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    data = {
        'Timestamp': pd.date_range(start='2023-01-01', periods=NUM_SAMPLES, freq='H'),
        'Shift': np.random.choice(shifts, NUM_SAMPLES),
        'Machine_Type': np.random.choice(machine_types, NUM_SAMPLES),
        'Material_Grade': np.random.choice(material_grades, NUM_SAMPLES),
        'Day_of_Week': np.random.choice(days, NUM_SAMPLES),
        
        # Numerical Columns
        'Injection_Temperature': np.random.normal(200, 10, NUM_SAMPLES),
        'Injection_Pressure': np.random.normal(100, 5, NUM_SAMPLES),
        'Cycle_Time': np.random.normal(30, 2, NUM_SAMPLES),
        'Cooling_Time': np.random.normal(10, 1, NUM_SAMPLES),
        'Material_Viscosity': np.random.normal(50, 5, NUM_SAMPLES),
        'Ambient_Temperature': np.random.normal(25, 2, NUM_SAMPLES),
        'Machine_Age': np.random.randint(1, 20, NUM_SAMPLES),
        'Operator_Experience': np.random.randint(1, 30, NUM_SAMPLES),
        'Maintenance_Hours': np.random.normal(5, 1, NUM_SAMPLES),
        'Machine_Utilization': np.random.uniform(0.7, 1.0, NUM_SAMPLES),
        'Temperature_Pressure_Ratio': np.random.normal(2, 0.1, NUM_SAMPLES),
        'Total_Cycle_Time': np.random.normal(40, 2, NUM_SAMPLES),
        'Efficiency_Score': np.random.uniform(0.8, 0.99, NUM_SAMPLES),
        
        # Target
        'Parts_Per_Hour': np.random.randint(50, 150, NUM_SAMPLES)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some NaN values to test imputation
    for col in ['Injection_Temperature', 'Shift', 'Machine_Utilization']:
        df.loc[df.sample(frac=0.01).index, col] = np.nan

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(FILE_PATH, index=False)
    print(f"Generated synthetic data at {FILE_PATH} with shape {df.shape}")

if __name__ == "__main__":
    generate_data()
