import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. Streamlit Page Config (MUST BE FIRST) ---
st.set_page_config(page_title="Glaucoma Diagnostic Assistant", page_icon="👁️")

# --- 2. Load the Trained Model (Cached) ---
@st.cache_resource
def load_model():
    model = joblib.load('glaucoma_model_final.pkl')
    model_features = joblib.load('model_features.pkl')
    return model, model_features

try:
    model, model_features = load_model()
except Exception:
    st.error("Model files not found. Please run the training script to generate .pkl files.")
    st.stop()

# --- 3. App Interface ---
st.title("👁️ AI Glaucoma Diagnostic Assistant")
st.markdown("""
This tool uses a **Clinically-Corrected Ensemble Model (XGBoost + Logistic Regression + Random Forest)** to assess Glaucoma risk.  
*Adjust the patient metrics in the sidebar to view the prediction.*
""")

# --- 4. Sidebar Inputs (Simulating Medical Devices) ---
st.sidebar.header("Patient Vitals")

# Key Biomarkers
iop = st.sidebar.slider(
    "Intraocular Pressure (IOP)", 10.0, 35.0, 18.0,
    help="Normal range is 12–22 mmHg"
)
cdr = st.sidebar.slider(
    "Cup-to-Disc Ratio (CDR)", 0.1, 0.9, 0.4,
    help="> 0.6 indicates risk"
)
rnfl = st.sidebar.slider(
    "OCT RNFL Thickness", 50.0, 120.0, 95.0,
    help="< 80 µm indicates nerve damage"
)
vf_sens = st.sidebar.slider("Visual Field Sensitivity", 0.0, 1.0, 0.85)

# Demographics
st.sidebar.subheader("Demographics & History")
age = st.sidebar.number_input("Age", 1, 100, 65)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
family_hist = st.sidebar.selectbox("Family History of Glaucoma", ["No", "Yes"])

# --- 5. Preprocessing Input ---
input_data = {feature: 0 for feature in model_features}

input_data.update({
    'Intraocular Pressure (IOP)': iop,
    'Cup-to-Disc Ratio (CDR)': cdr,
    'OCT_RNFL_Thickness': rnfl,
    'VF_Sensitivity': vf_sens,
    'Age': age,
    'Gender': 1 if gender == "Male" else 0,
    'Family History': 1 if family_hist == "Yes" else 0
})

df_input = pd.DataFrame([input_data])[model_features]

# --- 6. Visualization Functions ---
def plot_radar_chart(data):
    categories = ['IOP', 'Cup-to-Disc Ratio', 'Nerve Damage (RNFL)', 'Visual Field Loss']

    rnfl_score = (120 - data['OCT_RNFL_Thickness']) / 120
    vf_score = 1.0 - data['VF_Sensitivity']

    values = [
        data['Intraocular Pressure (IOP)'] / 30,
        data['Cup-to-Disc Ratio (CDR)'],
        rnfl_score,
        vf_score
    ]

    values += values[:1]
    categories += categories[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='red'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Patient Clinical Risk Profile"
    )
    return fig

# --- 7. Glaucoma Subtype Determination ---
def determine_subtype(row):
    if row['Age'] < 5:
        return "Congenital Glaucoma"
    elif row['Age'] < 40 and row['Intraocular Pressure (IOP)'] > 21:
        return "Juvenile Glaucoma"
    elif row['Intraocular Pressure (IOP)'] > 30:
        return "Angle-Closure Glaucoma"
    elif row['Intraocular Pressure (IOP)'] > 21 and row['OCT_RNFL_Thickness'] < 80:
        return "Primary Open-Angle Glaucoma"
    elif row['Intraocular Pressure (IOP)'] <= 21 and row['OCT_RNFL_Thickness'] < 80:
        return "Normal-Tension Glaucoma"
    elif row['Intraocular Pressure (IOP)'] > 21:
        return "Ocular Hypertension"
    else:
        return "Other / Unspecified"

# --- 8. Prediction ---
if st.button("Analyze Patient Data"):
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diagnostic Result")
        if prediction == 1:
            st.error("⚠️ **POSITIVE FOR GLAUCOMA**")
            subtype = determine_subtype(df_input.iloc[0])
            st.warning(f"Likely Subtype: **{subtype}**")
        else:
            st.success("✅ **NEGATIVE (Healthy)**")

    with col2:
        st.subheader("Model Confidence")
        st.metric("Probability", f"{probability:.1%}")

    st.info(f"""
    **Clinical Summary**
    - IOP: {iop} mmHg  
    - RNFL Thickness: {rnfl} µm  
    - CDR: {cdr}  

    *Risk increases when IOP > 21 mmHg and/or RNFL < 80 µm.*
    """)

    # Bar Chart
    fig = go.Figure(go.Bar(
        x=['IOP', 'CDR', 'RNFL Thickness', 'VF Sensitivity'],
        y=[iop, cdr, rnfl, vf_sens]
    ))
    fig.update_layout(title="Patient Metrics Overview")
    st.plotly_chart(fig)

    st.plotly_chart(plot_radar_chart(input_data))
