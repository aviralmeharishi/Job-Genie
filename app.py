import numpy as np
import pandas as pd
import streamlit as st
import pickle
import google.generativeai as genai

# Load the trained model
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Configure Gemini AI with Streamlit Secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

def generate_suggestions(prediction):
    """Generate AI-powered career suggestions using Gemini 2.0 Flash."""
    prompt = f"Provide career improvement generalised as well as personalised suggestions for a student with a placement probability of {prediction:.2f}. in both english and hindi and if probability is low make them feel motivated and if it is high prevent them from being overconfident"
    response = genai.GenerativeModel("gemini-2.0").generate_content(prompt)
    return response.text if response else "No suggestions available.


def predictor(inputs):
    inputs = np.array(inputs, dtype=object)
    pred = model.predict_proba([inputs])[0][1]
    
    suggestions = generate_suggestions(pred)
    
    if pred > 0.5:
        results = f"""✅ You Have Strong Chances of Getting Placed! Keep Up the Hard Work!\n
        Your Probability Of Getting Placed is {round(pred, 5)}\n
        AI Suggestions:\n{suggestions}\n
        Take Care!"""
        st.success(results)
    else:
        results = f"""⚠️ Your Placement Chances Are Low, But You Can Improve!\n
        Your Probability Of Getting Placed is {round(pred, 5)}\n
        AI Suggestions:\n{suggestions}\n
        Stay Positive & Keep Learning!"""
        st.error(results)
    
    return results


def chatbot():
    """Simple Gemini AI chatbot for career guidance."""
    st.subheader("🤖 Ask Job Genie (AI Chatbot)")
    user_input = st.text_input("Heyy , I am your Job Genie!!!!, Ask me anything about careers, skills, or jobs:")
    if st.button("Ask AI") and user_input:
        response = genai.GenerativeModel("gemini-2.0").generate_content(user_input)
        st.write("### 🤖 Job Genie AI Says:")
        st.write(response.text if response else "I'm unable to process your request.")


def main():
    st.title("Job Genie ✨'Your AI-Powered Path to Placement!'")
    st.markdown('### **AN AVIRAL MEHARISHI CREATION**')
    
    st.text('Job Genie is your AI-powered placement prediction tool...')
    
    cg = st.number_input('Enter Your CGPA Here:')
    ssc = st.number_input('Enter Your Senior Secondary Score Here:')
    hsc = st.number_input('Enter Your Higher Secondary Score Here:')
    pr = st.slider('Number of Projects:', min_value=0, max_value=10, step=1)
    inte = st.slider('Number of Internships:', min_value=0, max_value=6, step=1)
    wk = st.slider('Workshops/Certifications:', min_value=0, max_value=10, step=1)
    apt = st.number_input('Enter Your Aptitude Score:')
    ss = st.slider('Soft Skill Rating (Out of 5):', min_value=0.0, max_value=5.0, step=0.1)
    ext = 1 if st.radio('Are You in Extra Curricular Activities?', ['Yes', "No"]) == "Yes" else 0
    pt = 1 if st.radio('Attended Placement Training?', ['Yes', "No"]) == "Yes" else 0

    inputs = [cg, ssc, hsc, pr, inte, wk, apt, ss, ext, pt]
    
    if st.button('Show My Placement Prediction'):
        predictor(inputs)
    
    chatbot()
    
if __name__ == '__main__':
    main()
