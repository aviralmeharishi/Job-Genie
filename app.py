import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import pickle


with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)


def predictor(inputs):
    inputs = np.array(inputs, dtype= object)
    pred = model.predict_proba([inputs])[::1][0]


    if pred > 0.25:
        results = f'''✅ You Have Quite Strong Chances of Getting Placed in a Company So Maintain Your Consistency and Keep Working Hard
    Your Probability Of Getting Placed is {round(pred, 5)}
    Take Care!'''
        st.error(results)
        return results  # This ensures the function exits here
    
    else:
        result = f'''⚠️ You Have Quite Weak Chances of Getting Placed in a Company but don't worry You can achieve anything by just Working Hard
    Your Probability Of Getting Placed is {round(pred, 5)}
    Stay Healthy!'''
        st.success(result)
        return result




def main():

    st.title("Job Genie ✨'Your AI-Powered Path to Placement!'")

    st.markdown('### **AN AVIRAL MEHARISHI CREATION**')

    st.text('Job Genie is your AI-powered placement prediction tool that analyzes your responses to key questions and predicts your chances of getting hired. Whether you are a student or job seeker, Job Genie provides insights to help you improve your employability and prepare for success. Unlock your career potential with data-driven predictions! 🚀')

    cg = st.number_input('Enter Your CGPA Here :')

    ssc = st.number_input('Enter Your Senior Secondary Score Here :')

    hsc = st.number_input('Enter Your Higher Secondary Score Here :')

    pr = st.slider('How Many Projects Have You done Till now :', min_value= 0, max_value= 10, step = 1)

    inte = st.slider('How Many Internships Have You done Till now :', min_value= 0, max_value= 6, step = 1)

    wk = st.slider('How Many Workshops Have You attended Till now  or What About Your Number Of Certifications:', min_value= 0, max_value= 10, step = 1)

    apt = st.number_input('Enter Your Latest Aptitude Score Here :')

    ss = st.slider('Enter Your Soft Skill Rating (Out f 5)', min_value=0, max_value=5, step = 0.1)

    ext = (lambda x: 0 if x=='No' else 1)(st.radio('Are You Endulge in Any Extra Curricular Activities', ['Yes', "No"]))

    pt = (lambda x: 0 if x=='No' else 1)(st.radio('Have You Attended the Placement Training Session ', ['Yes', "No"]))


    inputs = [cg, ssc, hsc, pr, inte, wk, apt, ss, ext, pt]


    
    if st.button('Show Prediction of My Placemet'):
        response = predictor(inputs)


if __name__ == '__main__' :
    main()






    