import streamlit as st
import pandas as pd
import pickle
import os

# Constants
MODEL_PATH = "student_model.pkl"
GRADE_THRESHOLD_AT_RISK = 50
GRADE_THRESHOLD_AVERAGE = 75
MIN_STUDY_FOR_GOOD = 3
MAX_ABSENCES_FOR_GOOD = 5
MIN_PARTICIPATION = 3


@st.cache_resource
def load_model():
    """Load the trained ML model with error handling."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found!")
        st.stop()
    try:
        return pickle.load(open(MODEL_PATH, "rb"))
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


def classify_performance(prediction):
    """Classify grade into performance categories."""
    if prediction < GRADE_THRESHOLD_AT_RISK:
        return "At Risk"
    elif prediction < GRADE_THRESHOLD_AVERAGE:
        return "Average"
    return "High Performer"


st.title("AI Student Performance Prediction System")
st.write("Predict student final grades based on key performance indicators")

model = load_model()

# Input form
with st.form("student_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        study = st.number_input("Study Time (hours/day)", min_value=0, max_value=10, value=2)
        absences = st.number_input("Absences", min_value=0, max_value=30, value=3)
    
    with col2:
        previous = st.number_input("Previous Grade", min_value=0, max_value=100, value=60)
        assignments = st.number_input("Assignments Completed", min_value=0, max_value=10, value=5)
    
    participation = st.number_input("Participation Level (1-5)", min_value=1, max_value=5, value=3)
    
    submitted = st.form_submit_button("Predict Performance")

if submitted:
    input_data = pd.DataFrame({
        "StudyTime": [study],
        "Absences": [absences],
        "PreviousGrade": [previous],
        "AssignmentsCompleted": [assignments],
        "Participation": [participation]
    })
    
    try:
        prediction = model.predict(input_data)[0]
        category = classify_performance(prediction)
        
        st.metric("Predicted Final Grade", f"{prediction:.2f}")
        st.subheader(f"Category: {category}")
        
        # Recommendations
        st.write("### Recommendations")
        if study < MIN_STUDY_FOR_GOOD:
            st.warning("⚠️ Increase study time to improve performance")
        if absences > MAX_ABSENCES_FOR_GOOD:
            st.warning("⚠️ Improve attendance")
        if participation < MIN_PARTICIPATION:
            st.warning("⚠️ Increase classroom participation")
        if study >= MIN_STUDY_FOR_GOOD and absences <= MAX_ABSENCES_FOR_GOOD:
            st.success("✅ Great learning behavior — keep it up!")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")