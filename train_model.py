import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def main():
    print("Loading dataset...")
    df = pd.read_csv('diabetes.csv')

    print("Handling missing/zero values...")
    # Columns where 0 means missing value
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Replace 0 with NaN
    df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)
    
    # Fill NaN with median values
    for col in columns_with_zeros:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    print("Splitting data into features and target...")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    print("Creating 80/20 train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print(f"\nAccuracy Score: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

    print("Saving model and scaler...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("model.pkl and scaler.pkl saved successfully.")

if __name__ == "__main__":
    main()
