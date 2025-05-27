# Healthcare Insurance Fraud Detection

This project predicts potential healthcare fraud by analyzing provider, beneficiary, and medical claim (inpatient and outpatient) data. The notebook covers data loading, extensive preprocessing, feature engineering, model training, and evaluation, culminating in an XGBoost model optimized with GridSearchCV.

---

## 📊 Project Overview

The goal is to identify healthcare providers who are potentially committing fraud. This is achieved by combining multiple datasets related to beneficiaries, their medical claims, and the providers involved, and then applying machine learning classification models.

---

## 💾 Data Source

The data for this project is the "Medicare Provider Fraud Detection" dataset from Kaggle. It can be downloaded from:
[Kaggle: Medicare Provider Fraud Detection](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)

You will need the following files from the dataset:
* `Train_Beneficiarydata-1542865627584.csv`
* `Train-1542865627584.csv` (contains the target variable `PotentialFraud`)
* `Train_Inpatientdata-1542865627584.csv`
* `Train_Outpatientdata-1542865627584.csv`
* And corresponding test files (`Test-*.csv`, `Test_Beneficiarydata-*.csv`, etc.)

---

## ⚙️ Methodology & Key Steps

1.  **Data Loading & Initial Exploration:**
    * Multiple CSV files for training and testing are loaded, including beneficiary details, inpatient claims, outpatient claims, and the main training file with the `PotentialFraud` target.
    * Initial analysis of the target variable `PotentialFraud` in the `Train.csv` provider list shows an imbalance (No: 4904, Yes: 506).
    * The target variable is merged with inpatient and outpatient data to analyze fraud distribution; this analysis reveals fraud is more prevalent in inpatient claims.

2.  **Data Combination & Preprocessing:**
    * Inpatient and outpatient training data are combined, then merged with beneficiary data to create a comprehensive training set (`Train_All`). The combined dataset shows a less severe imbalance for `PotentialFraud` (No: 345415, Yes: 212796).
    * **Null Value Handling:**
        * Columns with over 50% null values are identified. Some are dropped, while others like physician details, admission/discharge dates, and diagnosis codes are kept for feature engineering or specific imputation.
        * NaNs in categorical columns (e.g., physician names, diagnosis codes) are filled with 'N/A'.
        * NaN `DeductibleAmtPaid` is filled with its median.
    * **Feature Engineering:**
        * `HospitalStay`: Calculated from `AdmissionDt` and `DischargeDt`; NaNs are set to 0.
        * Date columns (`ClaimStartDt`, `ClaimEndDt`, `DOB`) are converted to Unix timestamps.
        * `RenalDiseaseIndicator`: 'Y' is mapped to 1.
        * `Physician_coded`: A new feature encoding relationships between Attending, Operating, and Other physicians (e.g., if they are the same or different).
        * `Provider`: LabelEncoded.
    * Original date columns used for `HospitalStay` and other irrelevant identifiers like `BeneID`, `ClaimID` are dropped.

3.  **Model Preparation:**
    * The `train_data` is split into 80% training and 20% validation sets.
    * Non-numeric columns are identified and dropped before scaling.
    * `StandardScaler` is applied to the numerical features.
    * The target variable `PotentialFraud` ('Yes'/'No') is mapped to 1/0.

4.  **Model Training and Evaluation:**
    * Several baseline models are trained and evaluated on the validation set.
    * **XGBoost with GridSearchCV:** This is the best performing model, optimized for hyperparameters.

---

## 📈 Models & Results

The following models were evaluated, with key metrics on the validation set:

| Model                        | Accuracy | Precision (Yes) | Recall (Yes) | F1-score (Yes) | AUC   |
| :--------------------------- | :------- | :-------------- | :----------- | :------------- | :---- |
| Logistic Regression          | 0.63     | 0.58            | 0.11         | 0.18           | N/A   |
| Decision Tree Classifier     | 0.70     | 0.60            | 0.60         | 0.60           | N/A   |
| Random Forest Classifier     | 0.70     | 0.60            | 0.60         | 0.60           | N/A   |
| Gradient Boosting Classifier | 0.68     | 0.66            | 0.34         | 0.45           | N/A   |
| **XGBoost (GridSearchCV)** | **0.76** | **0.73** | **0.59** | **0.65** | **0.816** |

*Note: The Random Forest and initial XGBoost results in the notebook were based on predictions from the immediately preceding model's cell during the interactive development. The final XGBoost with GridSearchCV provides the most robust and optimized results.*

**Best XGBoost Parameters (from GridSearchCV):**
* `learning_rate`: 0.2
* `max_depth`: 5
* `n_estimators`: 300
* `subsample`: 1.0

---

