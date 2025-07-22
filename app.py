import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing tools
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load all necessary label encoders
encoders = {
    "workclass": joblib.load("workclass_encoder.pkl"),
    "marital-status": joblib.load("marital-status_encoder.pkl"),
    "occupation": joblib.load("occupation_encoder.pkl"),
    "relationship": joblib.load("relationship_encoder.pkl"),
    "race": joblib.load("race_encoder.pkl"),
    "gender": joblib.load("gender_encoder.pkl"),
    "native-country": joblib.load("native-country_encoder.pkl"),
    "income": joblib.load("income_encoder.pkl"),
}

# Page config
st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

# Input fields
age = st.sidebar.slider("Age", 18, 65, 30)
workclass = st.sidebar.selectbox("Workclass", encoders["workclass"].classes_)
fnlwgt = st.sidebar.number_input("Fnlwgt", value=189664, step=1) # Using mean as default
# Instead of educational-num, use education string and map it back
education_options = {
    1: 'Preschool', 2: '1st-4th', 3: '5th-6th', 4: '7th-8th',
    5: '9th', 6: '10th', 7: '11th', 8: '12th', 9: 'HS-grad',
    10: 'Some-college', 11: 'Assoc-voc', 12: 'Assoc-acdm',
    13: 'Bachelors', 14: 'Masters', 15: 'Prof-school', 16: 'Doctorate'
}
# Filter out the education levels that were removed during preprocessing
valid_education_options = {v: k for k, v in education_options.items() if v not in ['1st-4th', '5th-6th', 'Preschool']}
education = st.sidebar.selectbox("Education Level", list(valid_education_options.keys()))
educational_num = valid_education_options[education]


marital_status = st.sidebar.selectbox("Marital Status", encoders["marital-status"].classes_)
occupation = st.sidebar.selectbox("Occupation", encoders["occupation"].classes_)
relationship = st.sidebar.selectbox("Relationship", encoders["relationship"].classes_)
race = st.sidebar.selectbox("Race", encoders["race"].classes_)
gender = st.sidebar.selectbox("Gender", encoders["gender"].classes_)
capital_gain = st.sidebar.number_input("Capital Gain", value=0, step=100)
capital_loss = st.sidebar.number_input("Capital Loss", value=0, step=100)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", encoders["native-country"].classes_)


# Encode categorical inputs
encoded_workclass = encoders["workclass"].transform([workclass])[0]
encoded_marital_status = encoders["marital-status"].transform([marital_status])[0]
encoded_occupation = encoders["occupation"].transform([occupation])[0]
encoded_relationship = encoders["relationship"].transform([relationship])[0]
encoded_race = encoders["race"].transform([race])[0]
encoded_gender = encoders["gender"].transform([gender])[0]
encoded_native_country = encoders["native-country"].transform([native_country])[0]


# Create input dataframe with all features used in training
input_df = pd.DataFrame([[
    age,
    encoded_workclass,
    fnlwgt,
    educational_num,
    encoded_marital_status,
    encoded_occupation,
    encoded_relationship,
    encoded_race,
    encoded_gender,
    capital_gain,
    capital_loss,
    hours_per_week,
    encoded_native_country
]], columns=[
    'age',
    'workclass',
    'fnlwgt',
    'educational-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country'
])

# Scale input
scaled_input = scaler.transform(input_df)

st.subheader("Input Data")
st.write(input_df)

# Predict
if st.button("Predict Salary Class"):
    pred = model.predict(scaled_input)[0]
    result = encoders["income"].inverse_transform([pred])[0]
    st.success(f"Predicted Salary Class: {result}")

# Batch Prediction
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

def process_batch_prediction(batch_df, encoders, scaler, model):
    # Encode batch data - handle potential missing columns in uploaded CSV
    for col, encoder in encoders.items():
        if col in batch_df.columns:
            # Check if all values in the batch column exist in the encoder's classes
            if all(item in encoder.classes_ for item in batch_df[col].unique()):
                batch_df[col] = encoder.transform(batch_df[col])
            else:
                st.warning(f"Column '{col}' in uploaded CSV contains values not seen during training. These rows may not be predicted accurately.")
                # Option 1: Drop rows with unseen values
                # batch_df = batch_df[batch_df[col].isin(encoder.classes_)]
                # Option 2: Impute unseen values (e.g., with the mode)
                # mode_value = batch_df[col].mode()[0] # Or encoder.classes_[0]
                # batch_df[col] = batch_df[col].apply(lambda x: encoder.transform([mode_value])[0] if x not in encoder.classes_ else encoder.transform([x])[0])
                # For simplicity here, we'll just proceed and let the transform potentially raise errors or produce unexpected results for unseen values.
                # A more robust app would handle this more gracefully.
                try:
                    batch_df[col] = encoder.transform(batch_df[col])
                except ValueError as e:
                    st.error(f"Error encoding column '{col}': {e}. Please ensure your CSV contains valid values for this column.")
                    return None # Return None to indicate an error


    # Ensure all expected columns are present in the batch data before scaling
    expected_cols = scaler.feature_names_in_ # Assuming scaler has this attribute after fitting
    missing_cols = set(expected_cols) - set(batch_df.columns)
    for c in missing_cols:
        batch_df[c] = 0 # Add missing columns with a default value (e.g., 0)

    # Reorder columns to match training data
    batch_df = batch_df[expected_cols]

    # Scale batch data
    batch_scaled = scaler.transform(batch_df)

    # Predict
    batch_preds = model.predict(batch_scaled)
    batch_df['Predicted Salary'] = encoders["income"].inverse_transform(batch_preds)

    return batch_df


if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", batch_df)
    prediction_results = process_batch_prediction(batch_df, encoders, scaler, model)
    if prediction_results is not None:
        st.write("Prediction Results", prediction_results)
