import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def run_ml_pipeline(file_path):
    # 1. Load Data
    df = pd.read_csv(file_path)

    # 2. Preprocessing
    # Drop irrelevant columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Encode Gender (Binary)
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    # One-Hot Encode Geography
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

    # Define Features (X) and Target (y)
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # 3. Split Data into Training (80%) and Testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Feature Scaling (Crucial for Logistic Regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. Model Training & Evaluation
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"--- {name} ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))
        print("\n")


if __name__ == "__main__":
    run_ml_pipeline('data/Churn_Modelling.csv')