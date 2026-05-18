# Diabetes Prediction System

A complete Machine Learning web application built with Python and Streamlit that predicts whether a person is likely to have diabetes based on their medical parameters.

## Dataset

This project uses the Pima Indians Diabetes dataset from kaggle, which contains the following medical predictor variables:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

And one target variable, `Outcome` (0 = No diabetes, 1 = Diabetes).

## Project Structure

- `app.py`: Streamlit web application.
- `train_model.py`: Script to preprocess data, train the ML model, and save it.
- `requirements.txt`: Python dependencies.
- `download_data.py`: Script used to fetch the raw dataset and add column headers.
- `diabetes.csv`: The dataset used for training.
- `model.pkl`: The trained Logistic Regression model.
- `scaler.pkl`: The trained StandardScaler for feature scaling.

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
The pre-trained model and scaler are already saved in the repository (`model.pkl` and `scaler.pkl`). If you wish to retrain them, simply run:
```bash
python train_model.py
```

### 3. Run the Web Application
To start the Streamlit web application, run:
```bash
streamlit run app.py
```

The application will open in your default web browser, where you can enter medical parameters and get a real-time prediction for diabetes risk.
<img width="1897" height="929" alt="image" src="https://github.com/user-attachments/assets/6a0e7391-cc48-4242-9469-b0e9721a6360" />
