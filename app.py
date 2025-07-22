
# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

# Streamlit app setup
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on 5 numeric features.")

st.sidebar.header("📋 Input Employee Details")

# --- User Input Section (only 5 features) ---
age = st.sidebar.slider("Age", 18, 75, 30)
education = st.sidebar.number_input("Education (numeric)", min_value=0, max_value=16, value=12)
occupation = st.sidebar.number_input("Occupation (numeric code)", min_value=0, max_value=20, value=4)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
experience = st.sidebar.number_input("Experience (years)", min_value=0, max_value=50, value=5)

# --- Create Input DataFrame ---
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.write("### 🔎 Input Preview")
st.write(input_df)

# --- Prediction ---
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {'>50K' if prediction[0] == 1 else '≤50K'}")

# --- Batch Prediction Section ---
st.markdown("---")
st.subheader("📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with 5 columns: age, education, occupation, hours-per-week, experience", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        required_cols = ['age', 'education', 'occupation', 'hours-per-week', 'experience']

        if all(col in batch_data.columns for col in required_cols):
            st.write("📋 Uploaded File Preview")
            st.write(batch_data.head())

            prediction = model.predict(batch_data[required_cols])
            batch_data['Prediction'] = ['>50K' if p == 1 else '≤50K' for p in prediction]

            st.success("✅ Batch prediction completed!")
            st.write(batch_data)

            # Enable CSV download
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", csv, file_name="predicted_classes.csv", mime="text/csv")

        else:
            st.error(f"❌ Uploaded CSV must include only these columns: {', '.join(required_cols)}")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
