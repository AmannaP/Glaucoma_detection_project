import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# --- 1. Load the Trained Model ---
# (Ensure 'glaucoma_model_final.pkl' and 'model_features.pkl' are in the same folder)
try:
    model = joblib.load('glaucoma_model_final.pkl')
    model_features = joblib.load('model_features.pkl')
except:
    st.error("Model files not found. Please run the training script to generate .pkl files.")
    st.stop()

# --- 2. App Interface ---
st.set_page_config(page_title="Glaucoma Diagnostic Assistant", page_icon="👁️")

st.title("👁️ AI Glaucoma Diagnostic Assistant")
st.markdown("""
This tool uses a **Clinically-Corrected XGBoost Model** to assess Glaucoma risk.
*Adjust the patient metrics in the sidebar to view the prediction.*
""")

# --- 3. Sidebar Inputs (Simulating Medical Devices) ---
st.sidebar.header("Patient Vitals")

# Key Biomarkers
iop = st.sidebar.slider("Intraocular Pressure (IOP)", 10.0, 35.0, 18.0, help="Normal range is 12-22 mmHg")
cdr = st.sidebar.slider("Cup-to-Disc Ratio (CDR)", 0.1, 0.9, 0.4, help="> 0.6 indicates risk")
rnfl = st.sidebar.slider("OCT RNFL Thickness", 50.0, 120.0, 95.0, help="< 80 µm indicates damage")
vf_sens = st.sidebar.slider("Visual Field Sensitivity", 0.0, 1.0, 0.85)

# Other Features (We set defaults for the less critical ones for the demo)
st.sidebar.subheader("Demographics & History")
age = st.sidebar.number_input("Age", 18, 100, 65)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
family_hist = st.sidebar.selectbox("Family History of Glaucoma", ["No", "Yes"])

# --- 4. Preprocessing Input ---
# We must match the exact feature structure the model was trained on
input_data = {}

# Initialize all features to 0
for feature in model_features:
    input_data[feature] = 0

# Update with user inputs
input_data['Intraocular Pressure (IOP)'] = iop
input_data['Cup-to-Disc Ratio (CDR)'] = cdr
input_data['OCT_RNFL_Thickness'] = rnfl
input_data['VF_Sensitivity'] = vf_sens
input_data['Age'] = age
input_data['Gender'] = 1 if gender == "Male" else 0
input_data['Family History'] = 1 if family_hist == "Yes" else 0

# Convert to DataFrame
df_input = pd.DataFrame([input_data])
# --- 5. Visualization Functions ---
# Radar Chart for Patient Profile
def plot_radar_chart(input_data):
    # Normalize values to 0-1 scale for the chart (approximate max values)
    # IOP max ~30, CDR max 1.0, RNFL max 120 (inverted because lower is bad)
    
    categories = ['IOP', 'Cup-to-Disc Ratio', 'Nerve Damage (RNFL)', 'Visual Field Loss']
    
    # Invert RNFL and VF because Lower = Bad, but we want Bad = Outside of chart
    rnfl_score = (120 - input_data['OCT_RNFL_Thickness']) / 120
    vf_score = (1.0 - input_data['VF_Sensitivity']) 
    
    values = [
        input_data['Intraocular Pressure (IOP)'] / 30,  # Normalized IOP
        input_data['Cup-to-Disc Ratio (CDR)'],          # Already 0-1
        rnfl_score,
        vf_score
    ]
    
    # Close the loop for the chart
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Patient Profile',
        line_color='red'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Patient Clinical Risk Profile"
    )
    return fig


# Function to determine likely glaucoma subtype based on clinical rules (['Primary Open-Angle Glaucoma', 'Juvenile Glaucoma', 'Congenital Glaucoma','Normal-Tension Glaucoma','Angle-Closure Glaucoma', 'Secondary Glaucoma'])
def determine_subtype(row):
    if row['Intraocular Pressure (IOP)'] > 21 and row['OCT_RNFL_Thickness'] < 80:
        return "Primary Open-Angle Glaucoma"
    elif row['Intraocular Pressure (IOP)'] <= 21 and row['OCT_RNFL_Thickness'] < 80:
        return "Normal-Tension Glaucoma"
    elif row['Intraocular Pressure (IOP)'] > 21 and row['OCT_RNFL_Thickness'] >= 80:
        return "Ocular Hypertension"
    elif row['Age'] < 40 and row['Intraocular Pressure (IOP)'] > 21:
        return "Juvenile Glaucoma"
    elif row['Age'] < 5:
        return "Congenital Glaucoma"
    elif row['Intraocular Pressure (IOP)'] > 30:
        return "Angle-Closure Glaucoma"
    else:
        return "Other/Unspecified"

# --- 5. Prediction ---
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
            st.warning(f"⚠️ Likely Subtype: **{subtype}**")
        else:
            st.success("✅ **NEGATIVE (Healthy)**")
            
    with col2:
        st.subheader("Model Confidence")
        st.metric(label="Probability", value=f"{probability:.1%}")

    # Explainability (Why?)
    st.info(f"""
    **Clinical Logic:**
    - IOP: {iop} mmHg
    - RNFL Thickness: {rnfl} µm
    - CDR: {cdr}
    
    *High probability is driven by IOP > 21 and/or RNFL < 80.*
    """)
    # Visualization
    st.subheader("Patient Metrics Visualization")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['IOP', 'CDR', 'RNFL Thickness', 'VF Sensitivity'],
        y=[iop, cdr, rnfl, vf_sens],
        marker_color=['indianred', 'lightsalmon', 'lightseagreen', 'lightblue']
    ))
    fig.update_layout(title="Patient Metrics Overview", yaxis_title="Value")
    st.plotly_chart(fig)
    st.markdown("----")

    st.plotly_chart(plot_radar_chart(input_data))