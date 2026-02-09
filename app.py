import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# ===================== LOAD MODELS =====================
working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# ===================== SIDEBAR =====================
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# =====================================================
# ================= DIABETES PAGE =====================
# =====================================================
if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction using Hybrid ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Pregnancies')
        SkinThickness = st.text_input('Skin Thickness')
        DPF = st.text_input('Diabetes Pedigree Function')

    with col2:
        Glucose = st.text_input('Glucose Level')
        Insulin = st.text_input('Insulin Level')
        Age = st.text_input('Age')

    with col3:
        BP = st.text_input('Blood Pressure')
        BMI = st.text_input('BMI')

    if st.button('Diabetes Test Result'):
        user_input = [
            float(Pregnancies), float(Glucose), float(BP),
            float(SkinThickness), float(Insulin),
            float(BMI), float(DPF), float(Age)
        ]

        prediction = diabetes_model.predict([user_input])[0]

        try:
            risk = diabetes_model.predict_proba([user_input])[0][1] * 100
        except:
            risk = 100 if prediction == 1 else 0

        # ---------- HYBRID RULE OVERRIDE ----------
        if float(Glucose) > 160 and float(BMI) > 30:
            risk = max(risk, 75)

        st.subheader(f"Diabetes Risk Score: {risk:.2f}%")

        if risk > 60:
            st.error("High Risk of Diabetes")
            st.warning("⚠️ Diabetes can increase Heart and Kidney disease risk.")
        elif risk > 30:
            st.warning("Moderate Risk of Diabetes")
        else:
            st.success("Low Risk of Diabetes")

        if risk < 50:
            st.info("ℹ️ Low confidence prediction. Doctor consultation recommended.")

# =====================================================
# =============== HEART DISEASE PAGE ==================
# =====================================================
if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction using Hybrid ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
        trestbps = st.text_input('Resting Blood Pressure')
        restecg = st.text_input('Resting ECG')
        oldpeak = st.text_input('ST Depression')

    with col2:
        sex = st.text_input('Sex (1=Male, 0=Female)')
        chol = st.text_input('Cholesterol')
        thalach = st.text_input('Max Heart Rate')
        slope = st.text_input('Slope')

    with col3:
        cp = st.text_input('Chest Pain Type')
        fbs = st.text_input('Fasting Blood Sugar >120')
        exang = st.text_input('Exercise Induced Angina')
        ca = st.text_input('Major Vessels')
        thal = st.text_input('Thal (0,1,2)')

    if st.button('Heart Disease Test Result'):
        user_input = [
            float(age), float(sex), float(cp), float(trestbps),
            float(chol), float(fbs), float(restecg), float(thalach),
            float(exang), float(oldpeak), float(slope), float(ca), float(thal)
        ]

        prediction = heart_disease_model.predict([user_input])[0]

        try:
            risk = heart_disease_model.predict_proba([user_input])[0][1] * 100
        except:
            risk = 100 if prediction == 1 else 0

        # ---------- HYBRID RULE OVERRIDE ----------
        if float(age) > 45 and float(trestbps) > 140 and float(chol) > 240:
            risk = max(risk, 80)

        st.subheader(f"Heart Disease Risk Score: {risk:.2f}%")

        if risk > 60:
            st.error("High Risk of Heart Disease")
            st.warning("⚠️ Elevated cardiac risk may lead to stroke.")
        elif risk > 30:
            st.warning("Moderate Risk of Heart Disease")
        else:
            st.success("Low Risk of Heart Disease")

        if risk < 50:
            st.info("ℹ️ Low confidence prediction. ECG & clinical tests advised.")

# =====================================================
# ================= PARKINSONS PAGE ===================
# =====================================================
if selected == "Parkinsons Prediction":

    st.title("Parkinson's Disease Prediction using ML")

    cols = st.columns(5)
    inputs = []

    fields = [
        'Fo', 'Fhi', 'Flo', 'Jitter%', 'JitterAbs', 'RAP',
        'PPQ', 'DDP', 'Shimmer', 'Shimmer(dB)', 'APQ3',
        'APQ5', 'APQ', 'DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    for i, field in enumerate(fields):
        with cols[i % 5]:
            inputs.append(st.text_input(field))

    if st.button("Parkinson's Test Result"):
        user_input = [float(x) for x in inputs]

        prediction = parkinsons_model.predict([user_input])[0]

        try:
            risk = parkinsons_model.predict_proba([user_input])[0][1] * 100
        except:
            risk = 100 if prediction == 1 else 0

        st.subheader(f"Parkinson's Risk Score: {risk:.2f}%")

        if risk > 60:
            st.error("High Risk of Parkinson's Disease")
        elif risk > 30:
            st.warning("Moderate Risk of Parkinson's Disease")
        else:
            st.success("Low Risk of Parkinson's Disease")

        if risk < 50:
            st.info("ℹ️ Voice analysis confidence is low. Neurological evaluation recommended.")
