import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def train_default_model(data_path):
    # Load data
    df = pd.read_csv(data_path)

    # Assume 'defaulted' is the target column
    target = 'default'
    features = [col for col in df.columns if col != target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC-ROC Score:", roc_auc_score(y_test, y_prob))

    return model, scaler


def expected_loss(model, scaler, borrower_details, loan_amount, recovery_rate=0.1):
    """
    Predicts probability of default and calculates expected loss.
    :param model: Trained model
    :param scaler: Scaler used for training
    :param borrower_details: Dict of borrower's features
    :param loan_amount: Loan amount
    :param recovery_rate: Recovery rate (default: 10%)
    :return: Expected loss value
    """
    borrower_df = pd.DataFrame([borrower_details])
    borrower_scaled = scaler.transform(borrower_df)
    pd_default = model.predict_proba(borrower_scaled)[:, 1][0]
    loss_given_default = 1 - recovery_rate
    expected_loss_value = pd_default * loss_given_default * loan_amount

    return expected_loss_value


# Example usage:
if __name__ == "__main__":
    model, scaler = train_default_model("Task 3 and 4_Loan_Data.csv")
    new_borrower = {
        "customer_id": 12345,  # You can provide a dummy ID or leave it out if it's not needed
        "credit_lines_outstanding": 5,
        "loan_amt_outstanding": 10000,
        "total_debt_outstanding": 5000,
        "income": 50000,
        "years_employed": 5,
        "fico_score": 650
    }

    loan_amount = 20000
    loss = expected_loss(model, scaler, new_borrower, loan_amount)
    print("Expected Loss:", loss)
