import streamlit as st
import pickle
import numpy as np

# Load the saved models
with open("spam_detection_model.pkl", "rb") as file:
    nb_model = pickle.load(file)  # Naïve Bayes Model

with open("spam_detection_rf.pkl", "rb") as file:
    rf_model = pickle.load(file)  # Random Forest Model

# Streamlit UI
st.set_page_config(page_title="Email Spam Detector", page_icon="📩", layout="wide")

st.title("📩 Email Spam Detection")
st.write("Enter an email message to check if it's **Spam or Not Spam**.")

# Sidebar - Model Selection & Theme
st.sidebar.title("⚙ Settings")
model_choice = st.sidebar.radio("Select Model:", ["Naïve Bayes", "Random Forest"])
st.sidebar.markdown("---")

# Input Box for Email Text
email_text = st.text_area("✉️ Enter your email message:", height=150)

# Predict Button
if st.button("🔍 Detect Spam"):
    if email_text.strip():  # Ensure input is not empty
        # Convert input to list (needed for model prediction)
        email_input = [email_text]

        # Choose model based on user selection
        if model_choice == "Naïve Bayes":
            model = nb_model
        else:
            model = rf_model

        # Predict the result
        prediction = model.predict(email_input)[0]  # 0 = Not Spam, 1 = Spam
        probability = model.predict_proba(email_input)[0][1]  # Probability of Spam

        # Display results with styling
        st.subheader("🧐 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ **Spam Email!** (Confidence: {probability * 100:.2f}%)")
        else:
            st.success(f"✅ **Not Spam!** (Confidence: {(1 - probability) * 100:.2f}%)")
    else:
        st.warning("⚠️ Please enter an email message to analyze.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Stay Spam Free☠️")
