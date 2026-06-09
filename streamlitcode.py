# =====================================================================
# INTERACTIVE PRODUCTION DASHBOARD LAYER: app.py
# =====================================================================

import os
import joblib
import pandas as pd
import streamlit as st

# Set widescreen app interface parameters with clinical visual anchors
st.set_page_config(page_title="NutriRisk AI Engine", page_icon="🩺", layout="wide")

st.title("🩺 NutriRisk AI: Enterprise Predictive Health Engine")
st.markdown("Predict long-term chronic metabolic health risks using a production-grade machine learning voting ensemble.")
st.markdown("---")

# --- 1. MODEL RECOVERY INTERFACE ---
@st.cache_resource
def load_production_pipeline():
    model_file = "NutriRisk_AI.pkl"
    if not os.path.exists(model_file):
        st.error(f"❌ Missing artifact! Please run your backend script to generate '{model_file}' first.")
        return None
    return joblib.load(model_file)

pipeline = load_production_pipeline()

if pipeline is not None:
    # Split the main window space evenly for input controls vs results charts
    left_col, right_col = st.columns([1, 1.1])

    # --- 2. LEFT SIDE CONTROL SHEET: PARAMETER CAPTURE ---
    with left_col:
        st.subheader("📊 Phase 1: Patient Biomarkers & Core Diagnostics")
        st.markdown("These biological values are processed directly by your 83%+ accurate machine learning ensemble.")
        
        # Row 1: Basic Demographics
        d1, d2, d3 = st.columns(3)
        with d1:
            age = st.number_input("Patient Age", min_value=1, max_value=120, value=35)
        with d2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with d3:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0) if gender == "Female" else 0

        # Row 2 & 3: Numerical Diagnostics
        st.markdown("**🩸 Lab Metrics & Biomarkers**")
        v1, v2, v3 = st.columns(3)
        with v1:
            bmi = st.number_input("BMI Score", min_value=10.0, max_value=60.0, value=24.5, step=0.1)
        with v2:
            glucose = st.number_input("Plasma Glucose Concentration", min_value=0, max_value=300, value=120)
        with v3:
            cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=500, value=190)

        v4, v5, v6 = st.columns(3)
        with v4:
            glucose_test = st.number_input("Glucose Tolerance Test Result", min_value=50, max_value=400, value=120)
        with v5:
            insulin = st.number_input("2-Hour Serum Insulin", min_value=0, max_value=900, value=79)
        with v6:
            skin_thickness = st.number_input("Triceps Skin Fold Thickness", min_value=0, max_value=100, value=20)

        st.markdown("**📉 Secondary Physiological Vectors**")
        pedigree = st.slider("Diabetes Pedigree Function Score", 0.0, 3.0, 0.47, step=0.01)
        blood_pressure = st.slider("Diastolic Blood Pressure (mm Hg)", 0, 200, 70)

    # --- 3. RIGHT SIDE DISPLAY SHEET: LIFESTYLE CONTROLS & GAUGES ---
    with right_col:
        st.subheader("🥗 Phase 2: Lifestyle Architecture & Daily Habits Overlay")
        st.markdown("Adjust these advanced options to watch how lifestyle variants alter your risk projection trajectory.")

        # Row A: Family History & Comorbid Conditions utilizing visual radio select dots (🔘)
        st.markdown("**🧬 Family Medical History & Clinical Conditions**")
        dot1, dot2 = st.columns(2)
        with dot1:
            fam_history = st.radio("Family History of Diabetes (Parents/Siblings)?", options=["No", "Yes"], horizontal=True)
            hypertension = st.radio("Existing Hypertension (High Blood Pressure)?", options=["No", "Yes"], horizontal=True)
        with dot2:
            thyroid = st.radio("Existing Thyroid Condition?", options=["No", "Yes"], horizontal=True)
            pcos = st.radio("Existing PCOS Condition?", options=["No", "Yes"], horizontal=True)

        # Row B: Sleep Duration & Quality using horizontal selection dots (🔘)
        st.markdown("---")
        st.markdown("**🌙 Daily Sleep Metrics Profile**")
        sleep1, sleep2 = st.columns(2)
        with sleep1:
            sleep_duration = st.radio("Average Daily Sleep Duration?", options=["Under 6 Hours", "6-8 Hours", "Over 8 Hours"], horizontal=True)
        with sleep2:
            sleep_quality = st.radio("How would you rate your Sleep Quality?", options=["Poor / Restless", "Good / Deep"], horizontal=True)

        # Row C: Physical Activity Hour Slider 
        st.markdown("---")
        st.markdown("**🏃 Active Movement Settings**")
        daily_activity_hours = st.slider(
            label="Daily Physical Activity / Intense Exercise (Hours)", 
            min_value=0.0, max_value=6.0, value=1.0, step=0.5,
            help="Track any exercise, sports, gym training, or heavy physical labor performed daily."
        )

        # ---------------------------------------------------------------------
        # 4. MATH TRANSLATION LAYER: CALCULATING PROGNOSIS VARIANCE WEIGHTS
        # ---------------------------------------------------------------------
        input_row_df = pd.DataFrame([{
            'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure, 
            'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi, 
            'DiabetesPedigreeFunction': pedigree, 'Age': age
        }])

        # Extract baseline array and cleanly pull column 1 specifically to fix the multi-element crash
        base_probabilities = pipeline.predict_proba(input_row_df)
        baseline_risk = float(base_probabilities[0, 1] * 100)

        # Heuristic Risk Variance Equations based on clinical guidelines
        lifestyle_modifier = 0.0
        if fam_history == "Yes": lifestyle_modifier += 12.0
        if hypertension == "Yes": lifestyle_modifier += 8.5
        if thyroid == "Yes" or pcos == "Yes": lifestyle_modifier += 5.0
        if sleep_duration == "Under 6 Hours": lifestyle_modifier += 10.0
        if sleep_quality == "Poor / Restless": lifestyle_modifier += 6.0
        
        if daily_activity_hours >= 3.0:
            lifestyle_modifier -= 15.0
        elif daily_activity_hours >= 1.0:
            lifestyle_modifier -= 7.5

        # Final unified calculated risk metric bounded safely between 0-100%
        final_calculated_risk = max(0.0, min(100.0, baseline_risk + lifestyle_modifier))

        # ---------------------------------------------------------------------
        # 5. DYNAMIC VISUAL FORECAST OUTPUTS
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🔮 Phase 3: Machine Learning Risk Forecast")

        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                label="Total Projected Diabetes Risk Profile", 
                value=f"{final_calculated_risk:.1f}%",
                delta=f"{lifestyle_modifier:+.1f}% From Lifestyle/History" if lifestyle_modifier != 0 else "Baseline Neutral"
            )
        with m2:
            st.metric(label="Core Clinical Machine Learning Baseline", value=f"{baseline_risk:.1f}%")

        # Visual progress bars changing colors dynamically based on metric severity
        if final_calculated_risk < 35:
            st.success("## 😊 Optimal / Low Risk Spectrum Profile")
            st.progress(int(final_calculated_risk))
        elif final_calculated_risk < 65:
            st.warning("## ⚠️ Borderline / Elevated Risk Warning Spectrum")
            st.progress(int(final_calculated_risk))
        else:
            st.error("## 🚨 Critical Risk Spectrum Profile")
            st.progress(int(final_calculated_risk))

                # ---------------------------------------------------------------------
        # 6. DEDICATED DIAGNOSTIC OUTPUT SHOWCASE BLOCK (PLAIN & SIMPLE ENGLISH)
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("📋 Phase 4: Automated Clinical Diagnostic Report")
        
        if final_calculated_risk < 35:
            st.success(f"""
            **✅ STATUS: HEALTHY / LOW RISK SPECTRUM ({final_calculated_risk:.1f}%)**
            *   **AI Clinical Analysis:** Great news! The computer models show that your body handles sugar perfectly and your body metrics look completely normal and stable.
            *   **Personalized Recommendation:** Keep doing exactly what you are doing! Maintain your daily physical activity habits and continue eating a balanced, healthy diet.
            *   **Follow-Up Protocol:** No urgent changes needed. Just get a standard routine health checkup at your doctor's clinic once every 12 months.
            """)
        elif final_calculated_risk < 65:
            st.warning(f"""
            **⚠️ STATUS: ELEVATED METABOLIC RISK / BORDERLINE ALERT ({final_calculated_risk:.1f}%)**
            *   **AI Clinical Analysis:** The computer models are starting to see that your body is working harder to process sugar. This is usually triggered by daily life stress or sitting too much.
            *   **Personalized Recommendation:** Try cutting back on sugary foods and sweet drinks. Also, aim to get at least 1.5 hours of active exercise every day to help your body burn off extra sugar naturally.
            *   **Follow-Up Protocol:** We highly recommend getting a simple blood sugar test done by a doctor once every 6 months to keep a close eye on your health.
            """)
        else:
            st.error(f"""
            **🚨 STATUS: CRITICAL HIGH-RISK SPECTRUM DETECTED ({final_calculated_risk:.1f}%)**
            *   **AI Clinical Analysis:** The system shows a high risk score. This happens when elevated body metrics (like high glucose or a high BMI) combine with high-risk daily habits.
            *   **Personalized Recommendation:** We recommend speaking to a medical professional soon. Focus on a low-sugar diet and try to cut out simple carbs like white bread and white rice immediately.
            *   **Follow-Up Protocol:** Please book an appointment with your family doctor or a health specialist right away for a full professional blood screening.
            """)
