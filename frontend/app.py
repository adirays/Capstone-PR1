import streamlit as st
import pandas as pd
import joblib
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Manufacturing Predictor",
    page_icon="🏭",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align: center;'>🏭 Manufacturing Parts Per Hour Predictor</h1>
    <p style='text-align: center; color: gray;'>
    Predict production output using machine, process, and operational parameters
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ===============================
# LOAD MODEL FILES
# ===============================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model")

model = joblib.load(os.path.join(MODEL_PATH, "linear_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
feature_order = joblib.load(os.path.join(MODEL_PATH, "feature_order.pkl"))

# ===============================
# LAYOUT
# ===============================
left_col, right_col = st.columns(2)

# ===============================
# MACHINE & PROCESS INPUTS
# ===============================
with left_col:
    with st.expander("🔧 Machine & Process Parameters", expanded=True):
        Injection_Temperature = st.number_input("Injection Temperature (°C)", 150.0, 300.0, 220.0)
        Injection_Pressure = st.number_input("Injection Pressure (bar)", 40.0, 120.0, 75.0)
        Cycle_Time = st.number_input("Cycle Time (seconds)", 10.0, 120.0, 45.0)
        Cooling_Time = st.number_input("Cooling Time (seconds)", 5.0, 60.0, 18.0)
        Material_Viscosity = st.number_input("Material Viscosity", 100.0, 600.0, 320.0)
        Ambient_Temperature = st.number_input("Ambient Temperature (°C)", 15.0, 45.0, 30.0)
        Machine_Age = st.number_input("Machine Age (years)", 0, 30, 6)
        Operator_Experience = st.number_input("Operator Experience (years)", 0, 20, 4)
        Maintenance_Hours = st.number_input("Maintenance Hours", 0, 500, 120)

# ===============================
# OPERATIONAL INPUTS
# ===============================
with right_col:
    with st.expander("⚙️ Operational Settings", expanded=True):
        shift = st.selectbox("Shift", ["Morning", "Evening", "Night"])
        machine_type = st.selectbox("Machine Type", ["Type A", "Type B", "Type C"])
        material_grade = st.selectbox("Material Grade", ["Basic", "Standard", "Premium"])
        day = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]
        )

        st.markdown("**Performance Indicators**")
        Efficiency_Score = st.slider("Efficiency Score", 0.0, 1.0, 0.82)
        Machine_Utilization = st.slider("Machine Utilization", 0.0, 1.0, 0.78)

# ===============================
# DERIVED FEATURES
# ===============================
Temperature_Pressure_Ratio = Injection_Temperature / Injection_Pressure
Total_Cycle_Time = Cycle_Time + Cooling_Time

# ===============================
# BUILD INPUT DICTIONARY
# ===============================
input_dict = {
    "Injection_Temperature": Injection_Temperature,
    "Injection_Pressure": Injection_Pressure,
    "Cycle_Time": Cycle_Time,
    "Cooling_Time": Cooling_Time,
    "Material_Viscosity": Material_Viscosity,
    "Ambient_Temperature": Ambient_Temperature,
    "Machine_Age": Machine_Age,
    "Operator_Experience": Operator_Experience,
    "Maintenance_Hours": Maintenance_Hours,
    "Temperature_Pressure_Ratio": Temperature_Pressure_Ratio,
    "Total_Cycle_Time": Total_Cycle_Time,
    "Efficiency_Score": Efficiency_Score,
    "Machine_Utilization": Machine_Utilization,

    "Shift_Evening": 1 if shift == "Evening" else 0,
    "Shift_Night": 1 if shift == "Night" else 0,

    "Machine_Type_Type_B": 1 if machine_type == "Type B" else 0,
    "Machine_Type_Type_C": 1 if machine_type == "Type C" else 0,

    "Material_Grade_Premium": 1 if material_grade == "Premium" else 0,
    "Material_Grade_Standard": 1 if material_grade == "Standard" else 0,

    "Day_of_Week_Monday": 1 if day == "Monday" else 0,
    "Day_of_Week_Tuesday": 1 if day == "Tuesday" else 0,
    "Day_of_Week_Wednesday": 1 if day == "Wednesday" else 0,
    "Day_of_Week_Thursday": 1 if day == "Thursday" else 0,
    "Day_of_Week_Saturday": 1 if day == "Saturday" else 0,
    "Day_of_Week_Sunday": 1 if day == "Sunday" else 0,
}

st.divider()

# ===============================
# PREDICTION BUTTON
# ===============================
st.markdown("<h3 style='text-align:center;'>📊 Prediction</h3>", unsafe_allow_html=True)

if st.button("🚀 Predict Parts Per Hour", use_container_width=True):
    input_df = pd.DataFrame([[input_dict[f] for f in feature_order]], columns=feature_order)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"✅ **Predicted Parts Per Hour:** {prediction[0]:.2f}")
