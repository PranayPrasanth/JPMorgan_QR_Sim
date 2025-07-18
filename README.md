JPMorgan Chase Quantitative Research Simulation
This repository contains the solutions and analyses performed as part of the JPMorgan Chase Quantitative Research Virtual Experience. This simulation provided hands-on experience in various aspects of quantitative finance, including asset pricing, credit risk modeling, and statistical analysis, using Python.

Table of Contents
Project Overview

Tasks Completed

Task 1: Natural Gas Price Forecasting

Task 2: Gas Storage Contract Pricing

Task 3: Credit Default Prediction & Expected Loss

Task 4: FICO Score Bucketing Methodologies

Technologies Used

How to Run the Code

Contact

Project Overview
The JPMorgan Chase Quantitative Research Simulation is designed to expose participants to real-world challenges faced by quantitative analysts in a financial institution. This project involved developing analytical models and performing statistical analyses on simulated financial datasets to address problems in asset valuation and risk management.

Tasks Completed
Task 1: Natural Gas Price Forecasting
Objective: To forecast future natural gas prices using time series analysis.

Methodology:

Loaded and preprocessed historical natural gas price data, handling date conversions, setting monthly frequency, and interpolating missing values.

Developed and trained an ARIMA(2,1,2) model to capture trends and seasonality in the gas price series.

Implemented a function to estimate gas prices for both historical and future dates, with a specific focus on forecasting up to 12 months ahead.

Key Outcome: A robust time series forecasting model capable of predicting future gas prices based on historical patterns.

Task 2: Gas Storage Contract Pricing
Objective: To price a hypothetical natural gas storage contract.

Methodology:

Designed and implemented a Python function to simulate the cash flows associated with a gas storage contract.

The model accounts for various parameters including injection dates, withdrawal dates, market prices, injection/withdrawal rates, maximum storage capacity, and daily storage costs.

Calculated the total value of the contract by aggregating cash flows and subtracting storage costs.

Key Outcome: A functional model to value complex financial derivatives (gas storage contracts) based on market dynamics and operational constraints.

Task 3: Credit Default Prediction & Expected Loss
Objective: To build a model predicting loan default and calculate the Expected Loss (EL) for individual borrowers.

Methodology:

Loaded simulated loan data, identifying relevant features and the target variable (default).

Split the data into training and testing sets and applied StandardScaler for feature standardization.

Trained a Logistic Regression model to predict the Probability of Default (PD).

Developed a function to calculate Expected Loss (EL) using the formula: EL=PD
timesLGD
timesEAD (where LGD is Loss Given Default and EAD is Exposure at Default, represented by loan_amount).

Key Outcome: A predictive model for credit risk assessment and a framework for calculating expected financial losses on loans.

Task 4: FICO Score Bucketing Methodologies
Objective: To explore and compare different methodologies for bucketing FICO scores to enhance credit risk assessment.

Methodology:

Implemented two distinct approaches for FICO score bucketing:

MSE-Based Bucketing (K-Means Clustering): Utilized K-Means clustering to group FICO scores into N buckets based on minimizing intra-cluster variance.

Log-Likelihood Optimization: Developed an optimization function to determine bucket boundaries that maximize the log-likelihood of default probabilities within each bucket.

Assigned credit ratings based on the derived bucket boundaries.

Key Outcome: Comparative analysis of advanced statistical techniques for segmenting credit scores, providing insights into more granular risk categorization.

Technologies Used
Python: Core programming language for all tasks.

Pandas: Data manipulation and analysis.

NumPy: Numerical operations and array handling.

scikit-learn: Machine learning models (Logistic Regression, KMeans) and utilities (train_test_split, StandardScaler).

statsmodels: Time series analysis (ARIMA).

Matplotlib: Data visualization (e.g., for gas price forecasts).

SciPy: Scientific computing, specifically for optimization (scipy.optimize.minimize).

Jupyter Notebook: (Assumed environment for development and presentation).

How to Run the Code
Clone the repository:

git clone https://github.com/PranayPrasanth/PranayPrasanth.git
cd PranayPrasanth

Install dependencies:

pip install pandas numpy scikit-learn statsmodels matplotlib scipy

Place data files:

Ensure Nat_Gas (1).csv is in the root directory for Task 1 and Task 2.

Ensure Task 3 and 4_Loan_Data.csv is in the root directory for Task 3 and Task 4.

Run individual scripts:

For Task 1 (Natural Gas Forecasting): python task1_gas_forecasting.py (assuming you name the file)

For Task 2 (Gas Storage Pricing): python task2_gas_pricing.py

For Task 3 & 4 (Credit Risk): python task3_4_credit_risk.py

(Note: You might need to adjust the filenames in the main() functions or run them directly from a Jupyter Notebook if that was your primary development environment.)

Contact
Feel free to reach out if you have any questions or feedback.

LinkedIn: Pranay Prasanth's LinkedIn Profile

GitHub: PranayPrasanth's GitHub Profile
