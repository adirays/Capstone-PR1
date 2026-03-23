import streamlit as st
import pandas as pd
import requests
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Manufacturing Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# STYLE & HEADER
# ===============================
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A; /* Dark Blue */
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280; /* Gray */
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
    }
    </style>
    <div class="main-header">🏭 Manufacturing AI Dashboard</div>
    <div class="sub-header">Real-time Production Output Prediction & Analytics</div>
    """,
    unsafe_allow_html=True
)

# ===============================
# SIDEBAR: CONFIG & STATUS
# ===============================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Backend URL Configuration
    default_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    BACKEND_URL = st.text_input("Backend API URL", value=default_url, help="URL of the FastAPI backend")

    # Connection Check
    if st.button("🔄 Check Connection"):
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ Connected to Backend")
            else:
                st.warning(f"⚠️ Backend returned {response.status_code}")
        except requests.exceptions.RequestException:
            st.error("❌ Connection Failed")
    
    st.divider()
    st.info("ℹ️ **Tip:** Adjust parameters in the main tabs to see live predictions.")

# ===============================
# DATA FETCHING (STATS)
# ===============================
@st.cache_data(ttl=600)  # Cache for 10 mins
def get_historical_stats():
    try:
        response = requests.get(f"{BACKEND_URL}/dataset/summary", timeout=2)
        if response.status_code == 200:
            return response.json().get("summary", {})
    except:
        return {}
    return {}

stats = get_historical_stats()
avg_production = stats.get("Parts_Per_Hour", {}).get("mean", 0)

# ===============================
# MAIN LAYOUT: INPUTS
# ===============================
st.markdown("### 🛠️ Production Parameters")

# Organize inputs into tabs for a cleaner look
tab_machine, tab_process, tab_env = st.tabs(["🏗️ Machine Specs", "🔥 Process Control", "🌍 Environment & Ops"])

with tab_machine:
    col1, col2 = st.columns(2)
    with col1:
        Machine_Type = st.selectbox("Machine Type", ["Type A", "Type B", "Type C"])
        Machine_Age = st.slider("Machine Age (Years)", 0.0, 30.0, 6.0, help="Age of the injection molding machine")
    with col2:
        Maintenance_Hours = st.number_input("Maintenance Hours (Last Month)", 0, 500, 120)
        Operator_Experience = st.slider("Operator Experience (Years)", 0.0, 20.0, 4.0)

with tab_process:
    col1, col2, col3 = st.columns(3)
    with col1:
        Injection_Temperature = st.number_input("Injection Temp (°C)", 150.0, 300.0, 220.0)
        Injection_Pressure = st.number_input("Injection Pressure (bar)", 40.0, 120.0, 75.0)
    with col2:
        Cycle_Time = st.number_input("Cycle Time (sec)", 10.0, 120.0, 45.0)
        Cooling_Time = st.number_input("Cooling Time (sec)", 5.0, 60.0, 18.0)
    with col3:
        Material_Viscosity = st.number_input("Material Viscosity", 100.0, 600.0, 320.0)
        Material_Grade = st.selectbox("Material Grade", ["Basic", "Standard", "Premium"])

with tab_env:
    col1, col2 = st.columns(2)
    with col1:
        Shift = st.selectbox("Shift", ["Morning", "Evening", "Night"])
        Day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    with col2:
        Ambient_Temperature = st.slider("Ambient Temp (°C)", 15.0, 45.0, 30.0)
        Efficiency_Score = st.slider("Efficiency Score", 0.0, 1.0, 0.82)
        Machine_Utilization = st.slider("Machine Utilization", 0.0, 1.0, 0.78)

# Derived Features Calculation (Live)
Temperature_Pressure_Ratio = Injection_Temperature / Injection_Pressure if Injection_Pressure > 0 else 0
Total_Cycle_Time = Cycle_Time + Cooling_Time

# ===============================
# PREDICTION & VISUALIZATION
# ===============================
st.divider()

# Prepare Input Data
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

    "Shift_Evening": 1 if Shift == "Evening" else 0,
    "Shift_Night": 1 if Shift == "Night" else 0,

    "Machine_Type_Type_B": 1 if Machine_Type == "Type B" else 0,
    "Machine_Type_Type_C": 1 if Machine_Type == "Type C" else 0,

    "Material_Grade_Premium": 1 if Material_Grade == "Premium" else 0,
    "Material_Grade_Standard": 1 if Material_Grade == "Standard" else 0,

    "Day_of_Week_Monday": 1 if Day == "Monday" else 0,
    "Day_of_Week_Tuesday": 1 if Day == "Tuesday" else 0,
    "Day_of_Week_Wednesday": 1 if Day == "Wednesday" else 0,
    "Day_of_Week_Thursday": 1 if Day == "Thursday" else 0,
    "Day_of_Week_Saturday": 1 if Day == "Saturday" else 0,
    "Day_of_Week_Sunday": 1 if Day == "Sunday" else 0,
}

# Prediction Action
if st.button("🚀 Run Prediction Analysis", type="primary", use_container_width=True):
    with st.spinner("Analyzing parameters..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json={"features": input_dict},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", 0)
                
                # --- RESULTS SECTION ---
                st.markdown("### 📊 Analysis Results")
                
                # 1. Key Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Output", f"{prediction:.1f} Units/Hr", delta=f"{prediction - avg_production:.1f} vs Avg")
                m2.metric("Total Cycle Time", f"{Total_Cycle_Time:.1f} s", delta_color="inverse") # Lower is usually better?
                
                # Theoretical Max Calculation (3600 seconds in an hour / Total Cycle Time)
                # Note: Total_Cycle_Time is a derived feature which is Cycle + Cooling? 
                # Or just Cycle? Assuming Total Cycle Time for full production loop.
                theoretical_max = 3600 / Total_Cycle_Time if Total_Cycle_Time > 0 else 0
                efficiency_percent = (prediction / theoretical_max * 100) if theoretical_max > 0 else 0
                
                m3.metric("Theoretical Efficiency", f"{efficiency_percent:.1f}%", help="Prediction vs Physics Limit")

                # 2. Comparative Chart
                st.subheader("Performance Context")
                chart_data = pd.DataFrame({
                    "Category": ["Market Average", "Your Prediction", "Theoretical Max"],
                    "Parts Per Hour": [avg_production, prediction, theoretical_max]
                })
                
                # Simple Bar Chart
                st.bar_chart(chart_data.set_index("Category"), color="#3B82F6") # Blue color
                
                st.success("✅ Analysis Complete")
                
            else:
                st.error(f"❌ API Error: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(f"❌ Connection Error: Ensure Backend is running at `{BACKEND_URL}`")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

