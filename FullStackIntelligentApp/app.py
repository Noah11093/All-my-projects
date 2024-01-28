import pandas as pd
import numpy as np
import streamlit as st
import joblib

model = joblib.load("nn_model.joblib")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Full-Stack Intelligent App", page_icon="🕸️", layout="wide")

options_first_language = ['English', 'French', 'Other']
options_funding = ['Apprentice_PS ', 'GPOG_FT', 'Intl Offshore', 'Unknown', 
           'Intl Regular', 'Intl Transfer', 'Joint Program Ryerson', 
           'Joint Program UTSC', 'Second Career Program', 'Work Safety Insurance Board']
options_school = ['Advancement', 'Business', 'Communications', 'Community and Health', 'Hospitality', 'Engineering', 'Transportation']
options_fast_track = ['Yes', 'No']
options_coop = ['Yes', 'No']
options_residency = ['Domestic', 'International']
options_gender = ['Female', 'Male', 'Netural']
options_prev_edu = ['High School', 'Post Secondary']
options_age = ['0 to 18', '19 to 20', '21 to 25', '26 to 30', '31 to 35', '36 to 40', '41 to 50', '51 to 60', '61 to 65', '66+']
options_eng_grade = ['Level-130', 'Level-131', 'Level-140', 'Level-141', 'Level-150', 'Level-151', 'Level-160', 'Level-161', 'Level-170', 'Level-171', 'Level-180']

st.markdown("<h1 style='text-align: center; font-size: 2.5em;'>Full-Stack Intelligent App 🕸️</h1>", unsafe_allow_html=True)
    

def main():
    with st.form("road_traffic_severity_form"):
        
        st.subheader("Please enter the following inputs:")
        
        first_term_gpa = st.number_input("First Term GPA:", 0.0, 4.5, value=None)
        second_term_gpa = st.number_input("Second Term GPA:", 0.0, 4.5, value=None)
        first_language = st.radio("First Language:", options=options_first_language, index=None)
        funding = st.selectbox("Funding:", options=options_funding, index=None)
        school = st.selectbox("School:", options=options_school, index=None)
        fast_track = st.radio("Fast Track:", options=options_fast_track, index=None, key="1")
        coop = st.radio("Co-op:", options=options_coop, index=None, key="2")
        residency = st.radio("Residency:", options=options_residency, index=None, key="3")
        gender = st.radio("Gender:", options=options_gender, index=None, key="4")
        prev_edu = st.radio("Previous Education:", options=options_prev_edu, index=None, key="5")
        age = st.selectbox("Age Group:",options=options_age, index=None)
        avg_mark = st.number_input("High School Average Mark:", 0.0, 100.0, value=None, step=0.1, format="%1f")
        math_score = st.number_input("Math Score:", 0.0, 50.0, value=None, step=0.1, format="%1f")
        eng_grade = st.selectbox("English Grade:",options=options_eng_grade, index=None)
        
        submit = st.form_submit_button("Predict")
        
    if submit:
        input_arr = []
        cat_list = [options_first_language, options_funding, options_fast_track, options_coop,
                    options_residency, options_gender, options_prev_edu, options_age]
        
        cat_array = [first_language, funding, fast_track, coop, residency, gender, prev_edu, age]
        
        for i in range(len(cat_list)):
            encoder = list(range(1, len(cat_list[i]) + 1))
            for j in range(len(cat_list[i])):
                if cat_list[i][j] == cat_array[i]:
                    input_arr.append(encoder[j])
        num_arr = [first_term_gpa, second_term_gpa]
        pred_arr = np.array(num_arr + input_arr).reshape(1,-1)
        
        raw_prediction = model.predict(pred_arr)[0]
        binary_prediction = int(raw_prediction >= 0.5)

        st.write("Raw Prediction:", raw_prediction)
        st.write("Binary Prediction:", binary_prediction)

        if binary_prediction == 1:
            st.write("### The student is predicted to PERSIST in the program")
        else:
            st.write("### The student is predicted to DROPOUT from school")
            

if __name__ == '__main__':
    main()
