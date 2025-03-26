import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load both models
with open("spam_detection_model.pkl", "rb") as file:
    nb_model = pickle.load(file)

with open("spam_detection_rf.pkl", "rb") as file:
    rf_model = pickle.load(file)

# Function to predict spam or ham
def predict_email(email_text, model):
    return model.predict([email_text])[0]

# Sidebar
option = st.sidebar.radio("Select Option", ["Spam Detector", "Comparison"])

if option == "Spam Detector":
    st.title("📩 Spam Email Detector")
    email_text = st.text_area("Enter Email Text:", "")

    if st.button("Predict"):
        nb_result = predict_email(email_text, nb_model)
        rf_result = predict_email(email_text, rf_model)

        st.subheader("Predictions:")
        st.write(f"**Naïve Bayes Prediction:** {'Spam' if nb_result == 1 else 'Not Spam'}")
        st.write(f"**Random Forest Prediction:** {'Spam' if rf_result == 1 else 'Not Spam'}")

elif option == "Comparison":
    st.title("📊 Model Comparison")

    # Sample email dataset
    test_emails = [
        "Congratulations! You won a $1000 Walmart gift card. Claim now.",
        "Hey John, can you send me the project report by tomorrow?",
        "Exclusive offer just for you! Buy now and get 50% discount.",
        "Dear customer, your bank account is at risk. Click the link to verify.",
        "Let's catch up over coffee this weekend!"
    ]
    
    # Get predictions
    nb_preds = [predict_email(email, nb_model) for email in test_emails]
    rf_preds = [predict_email(email, rf_model) for email in test_emails]

    # Convert to DataFrame
    df = pd.DataFrame({
        "Emails": test_emails,
        "Naïve Bayes": ["Spam" if p == 1 else "Not Spam" for p in nb_preds],
        "Random Forest": ["Spam" if p == 1 else "Not Spam" for p in rf_preds]
    })
    
    st.write("### Model Predictions on Sample Emails")
    st.dataframe(df)

    # Count comparison
    nb_spam_count = sum(nb_preds)
    rf_spam_count = sum(rf_preds)
    
    # Bar chart comparison
    st.write("### Spam Detection Comparison")
    fig, ax = plt.subplots()
    models = ["Naïve Bayes", "Random Forest"]
    spam_counts = [nb_spam_count, rf_spam_count]

    ax.bar(models, spam_counts, color=["blue", "green"])
    ax.set_ylabel("Number of Emails Predicted as Spam")
    ax.set_title("Spam Count Comparison Between Models")
    st.pyplot(fig)

    # Pie Chart
    st.write("### Spam vs Not Spam Prediction (Random Forest)")
    rf_labels = ["Spam", "Not Spam"]
    rf_sizes = [rf_spam_count, len(test_emails) - rf_spam_count]

    fig2, ax2 = plt.subplots()
    ax2.pie(rf_sizes, labels=rf_labels, autopct="%1.1f%%", colors=["red", "yellow"], startangle=90)
    ax2.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig2)
