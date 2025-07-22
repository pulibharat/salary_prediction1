
# 💼 Salary Prediction Web App

This is a Streamlit-based web app that predicts whether a person's income is above or below 50K per year based on various features such as age, education, hours worked per week, etc.

## 🚀 Features

- Predict income using a trained Machine Learning model.
- Select only a few important features or all features for prediction.
- User-friendly interface built with Streamlit.
- Model trained using the UCI Adult Income Dataset.
- Encoded categorical features with LabelEncoder.
- Deployed online using Streamlit Community Cloud.

## 🧠 Technologies Used

- **Python**
- **Pandas**
- **Scikit-learn**
- **Streamlit**
- **Joblib**

## 🏗️ How It Works

1. The user inputs personal data like age, education level, and work hours.
2. The app uses a trained ML model to predict whether their income is `>50K` or `<=50K`.
3. The result is displayed instantly on the web interface.

## 🗂️ Files

| File Name | Description |
|-----------|-------------|
| `app.py` | Main Streamlit app file |
| `model.pkl` | Trained machine learning model |
| `income_encoder.pkl` | LabelEncoder for encoding income |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation (this file) |

## 🔮 Sample Prediction Features

- Age  
- Workclass  
- Education  
- Occupation  
- Hours-per-week  
- Capital Gain  
- Capital Loss  

## 🧪 Model Training

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
