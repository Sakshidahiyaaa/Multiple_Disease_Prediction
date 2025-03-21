import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# 🔹 Load the saved models **before checking if they are fitted**
try:
    diabetes_model = pickle.load(open(os.path.join(working_dir, 'trained_model.sav'), 'rb'))
    heart_disease_model = pickle.load(open(os.path.join(working_dir, 'heart_disease_model.sav'), 'rb'))
    parkinsons_model = pickle.load(open(os.path.join(working_dir, 'parkinsons_model.sav'), 'rb'))
except FileNotFoundError as e:
    st.error(f"🚨 Model file not found: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()

# 🔹 Function to check if the model is fitted
def is_model_fitted(model):
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False

# 🔹 Check if models are trained before using them
if not is_model_fitted(diabetes_model):
    st.error("🚨 Diabetes model is NOT trained. Train it in Jupyter Notebook and re-save it!")
    st.stop()

if not is_model_fitted(heart_disease_model):
    st.error("🚨 Heart Disease model is NOT trained. Train it in Jupyter Notebook and re-save it!")
    st.stop()

if not is_model_fitted(parkinsons_model):
    st.error("🚨 Parkinson's model is NOT trained. Train it in Jupyter Notebook and re-save it!")
    st.stop()

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Helper function to safely convert user inputs to float
def safe_float_convert(inputs):
    try:
        return [float(x) if x.strip() else 0.0 for x in inputs]
    except ValueError:
        st.error("Invalid input detected. Please enter numeric values only.")
        return []

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.text_input('Number of Pregnancies')
    with col2: Glucose = st.text_input('Glucose Level')
    with col3: BloodPressure = st.text_input('Blood Pressure value')
    with col1: SkinThickness = st.text_input('Skin Thickness value')
    with col2: Insulin = st.text_input('Insulin Level')
    with col3: BMI = st.text_input('BMI value')
    with col1: DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2: Age = st.text_input('Age of the Person')

    if st.button('Diabetes Test Result'):
        user_input = safe_float_convert([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                                         BMI, DiabetesPedigreeFunction, Age])
        if user_input:
            diab_prediction = diabetes_model.predict([user_input])
            st.success('The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic')

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1: age = st.text_input('Age')
    with col2: sex = st.text_input('Sex')
    with col3: cp = st.text_input('Chest Pain types')
    with col1: trestbps = st.text_input('Resting Blood Pressure')
    with col2: chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3: fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1: restecg = st.text_input('Resting Electrocardiographic results')
    with col2: thalach = st.text_input('Maximum Heart Rate achieved')
    with col3: exang = st.text_input('Exercise Induced Angina')
    with col1: oldpeak = st.text_input('ST depression induced by exercise')
    with col2: slope = st.text_input('Slope of the peak exercise ST segment')
    with col3: ca = st.text_input('Major vessels colored by flourosopy')
    with col1: thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    if st.button('Heart Disease Test Result'):
        user_input = safe_float_convert([age, sex, cp, trestbps, chol, fbs,
                                         restecg, thalach, exang, oldpeak, slope, ca, thal])
        if user_input:
            heart_prediction = heart_disease_model.predict([user_input])
            st.success('The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease')

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: fo = st.text_input('MDVP:Fo(Hz)')
    with col2: fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3: flo = st.text_input('MDVP:Flo(Hz)')
    with col4: Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5: Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col1: RAP = st.text_input('MDVP:RAP')
    with col2: PPQ = st.text_input('MDVP:PPQ')
    with col3: DDP = st.text_input('Jitter:DDP')
    with col4: Shimmer = st.text_input('MDVP:Shimmer')
    with col5: Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    with col1: APQ3 = st.text_input('Shimmer:APQ3')
    with col2: APQ5 = st.text_input('Shimmer:APQ5')
    with col3: APQ = st.text_input('MDVP:APQ')
    with col4: DDA = st.text_input('Shimmer:DDA')
    with col5: NHR = st.text_input('NHR')
    with col1: HNR = st.text_input('HNR')
    with col2: RPDE = st.text_input('RPDE')
    with col3: DFA = st.text_input('DFA')
    with col4: spread1 = st.text_input('spread1')
    with col5: spread2 = st.text_input('spread2')
    with col1: D2 = st.text_input('D2')
    with col2: PPE = st.text_input('PPE')

    if st.button("Parkinson's Test Result"):
        user_input = safe_float_convert([fo, fhi, flo, Jitter_percent, Jitter_Abs,
                                         RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                                         APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE])
        if user_input:
            parkinsons_prediction = parkinsons_model.predict([user_input])
            st.success("The person has Parkinson's disease" if parkinsons_prediction[0] == 1 
                       else "The person does not have Parkinson's disease")
