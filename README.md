# 🧮 JPMorgan Chase Quantitative Research Simulation

This repository contains the solutions and analyses performed as part of the **JPMorgan Chase Quantitative Research Virtual Experience**. The simulation provided hands-on experience in various aspects of **quantitative finance**, including asset pricing, credit risk modeling, and statistical analysis using Python.

---

## 📑 Table of Contents

- [Project Overview](#project-overview)  
- [Tasks Completed](#tasks-completed)  
  - [Task 1: Natural Gas Price Forecasting](#task-1-natural-gas-price-forecasting)  
  - [Task 2: Gas Storage Contract Pricing](#task-2-gas-storage-contract-pricing)  
  - [Task 3: Credit Default Prediction & Expected Loss](#task-3-credit-default-prediction--expected-loss)  
  - [Task 4: FICO Score Bucketing Methodologies](#task-4-fico-score-bucketing-methodologies)  
- [Technologies Used](#technologies-used)  
- [How to Run the Code](#how-to-run-the-code)  
- [Contact](#contact)  

---

## Project Overview

The **JPMorgan Chase Quantitative Research Simulation** is designed to expose participants to real-world challenges faced by quantitative analysts in a financial institution. This project involved developing analytical models and performing statistical analyses on simulated financial datasets to address problems in asset valuation and risk management.

---

## Tasks Completed

### Task 1: Natural Gas Price Forecasting

**Objective:** Forecast future natural gas prices using time series analysis.

**Methodology:**
- Loaded and preprocessed historical natural gas price data, handling date conversions and missing values.
- Trained an **ARIMA(2,1,2)** model to capture trends and seasonality.
- Forecasted gas prices up to 12 months into the future.

**Key Outcome:** A robust time series forecasting model capable of predicting future gas prices.

---

### Task 2: Gas Storage Contract Pricing

**Objective:** Price a hypothetical natural gas storage contract.

**Methodology:**
- Simulated cash flows for injection and withdrawal periods.
- Considered parameters like market prices, storage limits, injection/withdrawal rates, and daily storage costs.
- Aggregated cash flows and costs to value the contract.

**Key Outcome:** A working financial model to value gas storage derivatives under operational constraints.

---

### Task 3: Credit Default Prediction & Expected Loss

**Objective:** Build a model to predict loan default and calculate Expected Loss (EL).

**Methodology:**
- Preprocessed and standardized simulated loan data.
- Trained a **Logistic Regression** model to estimate Probability of Default (PD).
- Calculated **Expected Loss (EL)** using the formula:  
  `EL = PD × LGD × EAD`

**Key Outcome:** A predictive model for credit risk assessment and expected loss calculation.

---

### Task 4: FICO Score Bucketing Methodologies

**Objective:** Explore and compare methodologies for FICO score bucketing.

**Methodology:**
- **MSE-Based Bucketing:** Used K-Means clustering to minimize intra-cluster variance.
- **Log-Likelihood Optimization:** Used `scipy.optimize.minimize` to optimize bucket boundaries based on log-likelihood of defaults.
- Assigned credit ratings to buckets for risk categorization.

**Key Outcome:** Comparative analysis of credit score segmentation for improved risk stratification.

---

## Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **scikit-learn**: Machine learning (Logistic Regression, KMeans)
- **statsmodels**: Time series (ARIMA)
- **Matplotlib**: Visualizations
- **SciPy**: Optimization algorithms
- **Jupyter Notebook**: Development and presentation

---

## How to Run the Code

1. **Clone the repository**  
```bash
git clone https://github.com/PranayPrasanth/PranayPrasanth.git
cd PranayPrasanth
```

2. **Install dependencies**  
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib scipy
```

3. **Place data files in root directory**  
- `Nat_Gas (1).csv` → for Task 1 and Task 2  
- `Task 3 and 4_Loan_Data.csv` → for Task 3 and Task 4  

4. **Run individual scripts**  
```bash
python task1_gas_forecasting.py
python task2_gas_pricing.py
python task3_4_credit_risk.py
```

> *Alternatively, open and run the corresponding Jupyter Notebooks for each task.*

---

## Contact

- **LinkedIn:** [Pranay Prasanth](https://www.linkedin.com/in/pranayprasanth/)  
- **GitHub:** [PranayPrasanth](https://github.com/PranayPrasanth)  
