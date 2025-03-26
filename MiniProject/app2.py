import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
option = st.sidebar.radio("Select Option:", ["Single Model Prediction", "Comparison"])

if option == "Single Model Prediction":
    model_choice = st.sidebar.radio("Select Model:", ["Naïve Bayes", "Random Forest"])

st.sidebar.markdown("---")

# Input Box for Email Text
email_text = st.text_area("✉️ Enter your email message:", height=150)

# Predict Button
if st.button("🔍 Detect Spam"):
    if email_text.strip():  # Ensure input is not empty
        email_input = [email_text]

        if option == "Single Model Prediction":
            model = nb_model if model_choice == "Naïve Bayes" else rf_model
            prediction = model.predict(email_input)[0]
            probability = model.predict_proba(email_input)[0][1]

            st.subheader("🧐 Prediction Result")
            if prediction == 1:
                st.error(f"⚠️ **Spam Email!** (Confidence: {probability * 100:.2f}%)")
            else:
                st.success(f"✅ **Not Spam!** (Confidence: {(1 - probability) * 100:.2f}%)")


        elif option == "Comparison":
            # Predictions from both models
            nb_pred = nb_model.predict(email_input)[0]
            nb_prob = nb_model.predict_proba(email_input)[0][1]

            rf_pred = rf_model.predict(email_input)[0]
            rf_prob = rf_model.predict_proba(email_input)[0][1]

            # Convert Spam Probability to Confidence for Consistency
            nb_confidence = (1 - nb_prob) * 100
            rf_confidence = (1 - rf_prob) * 100

            # Prepare Data for Visualization
            data = pd.DataFrame({
            "Model": ["Naïve Bayes", "Random Forest"],
            "Spam Probability (%)": [nb_prob * 100, rf_prob * 100],  # Raw Spam Probability
            "Confidence (%)": [nb_confidence, rf_confidence],  # Confidence of Not Spam
            "Prediction": ["Spam" if nb_pred == 1 else "Not Spam", "Spam" if rf_pred == 1 else "Not Spam"]
            })

            # Display results in table
            st.table(data)


            # Bar Chart for Probability Comparison
            fig, ax = plt.subplots()
            ax.bar(data["Model"], data["Spam Probability (%)"], color=['blue', 'green'])  # Use correct column name
            ax.set_ylabel("Spam Probability (%)")
            ax.set_title("Model Comparison")
            st.pyplot(fig)

    else:
        st.warning("⚠️ Please enter an email message to analyze.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Stay Spam Free☠️")
