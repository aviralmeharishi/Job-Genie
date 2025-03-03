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
    """Generate AI-powered career suggestions using Gemini 1.5 Flash."""
    model = genai.GenerativeModel("gemini-1.5-flash")  # ✅ Using a supported model
    
    prompt = f"""
    Provide career improvement suggestions in both English and Hindi for a student with a placement probability of {prediction:.2f}.
    
    - If the probability is low, provide motivational advice to boost their confidence and encourage them.
    - If the probability is high, give guidance to prevent overconfidence and suggest ways to continue improving.
    - Include both general career tips and personalized advice based on the probability.
    """
    
    response = model.generate_content(prompt)  # Generate AI response
    
    if response and hasattr(response, "text"):
        return response.text  # Extract text response
    else:
        return "No suggestions available at the moment."

def predictor(inputs):
    inputs = np.array(inputs, dtype=object)
    pred = model.predict_proba([inputs])[0][1]
    
    suggestions = generate_suggestions(pred)
    
    if pred > 0.5:
        results = f"""✅ You Have Strong Chances of Getting Placed! Keep Up the Hard Work! 🚀\n
        **Your Probability Of Getting Placed:** {round(pred, 5)}\n
        **AI Suggestions:**\n{suggestions}\n
        Stay focused, and best of luck! 🎯"""
        st.success(results)
    else:
        results = f"""⚠️ Your Placement Chances Are Low, But You Can Improve! 💡\n
        **Your Probability Of Getting Placed:** {round(pred, 5)}\n
        **AI Suggestions:**\n{suggestions}\n
        Don't worry! Keep learning and growing! 🌱"""
        st.error(results)
    
    return results

def chatbot():
    """AI-powered chatbot for career guidance using Gemini 1.5 Flash."""
    st.subheader("🤖 Job Genie - Your AI Career Coach!")
    st.write("👋 **Hi, I am Job Genie!** Your personal AI career assistant. Ask me anything about jobs, skills, career growth, or placements, and I'll guide you! 🚀")

    user_input = st.text_input("💬 What's on your mind? (e.g., 'How can I improve my resume?' or 'What skills do I need for Data Science?')")

    if st.button("🔮 Ask Job Genie"):
        if user_input:
            model = genai.GenerativeModel("gemini-1.5-flash")  # ✅ Using a supported model
            response = model.generate_content(user_input)  # Get AI response

            if response and hasattr(response, "text"):
                st.write("### 🤖 Job Genie Says:")
                st.write(response.text)
            else:
                st.write("⚠️ Oops! I couldn't generate a response. Please try again.")
        else:
            st.warning("🚀 Please enter a question before clicking 'Ask Job Genie'.")

def main():
    st.title("✨ Job Genie - Your AI-Powered Path to Placement!")
    st.markdown('### **AN AVIRAL MEHARISHI CREATION** 🎓')

    st.write("👋 **Hey there! I’m Job Genie!** I'm here to predict your placement chances and guide you on your career journey! 🌟")

    st.markdown("### 📌 **Let's Get Started!** Fill in your details below:")

    cg = st.number_input('🎓 **Enter Your CGPA:**')
    ssc = st.number_input('📖 **Senior Secondary Score (10th %):**')
    hsc = st.number_input('📚 **Higher Secondary Score (12th %):**')
    pr = st.slider('💻 **Number of Projects Completed:**', min_value=0, max_value=10, step=1)
    inte = st.slider('🛠️ **Number of Internships Done:**', min_value=0, max_value=6, step=1)
    wk = st.slider('📜 **Workshops/Certifications Attended:**', min_value=0, max_value=10, step=1)
    apt = st.number_input('🧠 **Enter Your Aptitude Score:**')
    ss = st.slider('💬 **Soft Skill Rating (Out of 5):**', min_value=0.0, max_value=5.0, step=0.1)
    ext = 1 if st.radio('🎭 **Are You Active in Extra Curricular Activities?**', ['Yes', "No"]) == "Yes" else 0
    pt = 1 if st.radio('📈 **Have You Attended Placement Training?**', ['Yes', "No"]) == "Yes" else 0

    inputs = [cg, ssc, hsc, pr, inte, wk, apt, ss, ext, pt]
    
    if st.button('✨ Show My Placement Prediction'):
        predictor(inputs)
    
    chatbot()
    
if __name__ == '__main__':
    main()
