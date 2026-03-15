# 🔍 Fraud Detection — Supervised Learning Classification Models
### PaySim Mobile Money Fraud Detection | Group 12 Capstone Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458?style=flat-square&logo=pandas)
![Kaggle](https://camo.githubusercontent.com/c4378bfd53be39b991aae644ae11f441ec46a2d916c32fd622d14726ccfad6cd/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4b6167676c652d3230424546463f7374796c653d666c6174266c6f676f3d6b6167676c65266c6f676f436f6c6f723d7768697465)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)

## GROUP 12 ACTIVE CONTRIBUTORS

### Group Leader

| Name | Email | GitHub | Role |
|---|---|---|---|
| Princess Chiamaka Emenari | princessemenari2@gmail.com | | Group Leader |

---

### Active Members

| Name | Email | GitHub | Role |
|---|---|---|---|
| Kolawole Julius Oluwatobi | anthonyjk1204@gmail.com | [GitHub](https://github.com/kjuls) | Active Member |
| Olaleru Praise Ajibola | zionpraiz9@gmail.com | [GitHub](https://github.com/zionpraiz9-code/TS_ACADEMY_CAPSTONE-PROJECT-GROUP-12/tree/main) | Active Member |
| Adeleye Adekunle Oluwaseun | dequnle7@gmail.com | [GitHub](https://github.com/qunlecrown/TS-ACADEMY-GROUP-12-CAPSTONE-PROJECT/tree/main) | Active Member |
| Udoh Edidiong Monday | beeeddy22@gmail.com | [GitHub](https://github.com/Edidiong-Udoh2/CAPSTONE-GROUP-12-PROJECT) | Active Member |
| Ukonu Fortune Chiemela | ukonufortune@gmail.com | [GitHub](https://github.com/Fortuneukonu) | Active Member |
| Ogunniyi Ibrahim Adedeji | ogunniyiibrahim2029@gmail.com | [GitHub](https://github.com/nobleXibrahim/fraud-detection) | Active Member |
| Titus Oluwafemi Ojo | femititus@gmail.com | [GitHub](https://github.com/femititus/TS_ACADEMY_GROUP_12_CAPSTONE_PROJECT_FRAUD_DETECTION_2) | Active Member |


## 📌 Table of Contents

- [Project Overview]
- [Dataset]
- [Project Structure]
- [Methodology]
  - [Stage 1 — Data Cleaning & Preparation]
  - [Stage 2 — Data Distribution]
  - [Stage 3 — Bivariate & Multivariate Analysis]
  - [Stage 4 — Data Preprocessing]
  - [Stage 5 — Machine Learning]
- [Key Findings]
- [Model Performance]
- [Feature Importance]
- [Results Summary]
- [Recommendations]


## PROJECT OVERVIEW

Financial fraud continues to be one of the most significant threats to digital banking systems. As the volume of electronic transactions grows, detecting fraudulent activity quickly and accurately has become essential for financial institutions. This project develops a machine learning–based fraud detection system that analyzes transaction behavior and identifies suspicious activity in real time. By leveraging historical transaction data and behavioral patterns, the model learns to distinguish between legitimate and fraudulent transactions. The project applies a complete end-to-end machine learning workflow, including data preprocessing, feature engineering, handling class imbalance, model training, and performance evaluation.
Two machine learning Models were implemented and compared:
Logistic Regresiion
Random Forest Classifier

To address the common challenge of imbalanced fraud datasets, undersampling techniques were applied to ensure the model could effectively learn fraud patterns. Key transaction features such as transaction amount, transaction frequency, account balance behavior, and recent transaction patterns were used to train the models. The results demonstrate how machine learning can significantly improve fraud detection capabilities by identifying high-risk transactions while minimizing false positives. This project highlights how data-driven models can support financial institutions in preventing fraud, protecting customers, and improving transaction security.

### PROJECT OBJECTIVES

The major goal of this capstone project is to build classification models that could be able to identify the patterns in the various transactions and train these models in order to predict the outcome of the transactions if they are fraud transactions or not. These models are built in an attempt to be deployed into real world data especially financial databases and be able to use the models that we build and train to immediately flag transactions who follows the same pattern of the transactions flagged as fraud and prevent future users from falling victims of fraud.

#### Specific Objectives

- Detect fraudulent financial transactions using machine learning
- Analyze transaction behavior and spending patterns
- Handle imbalanced fraud datasets using resampling techniques
- Train and evaluate classification models for fraud
- Identify key transaction features that contribute to fraud detection

## DATA SOURCE AND JUSTIFICATION
The dataset being used is a fraud detection dataset of users who carried out transactions using paysim. PaySim is a financial simulator that simulates mobile money transactions based on an original dataset. Although the dataset was generated synthetically using paysim, this dataset was chosen because of how good it relates with real world financial transactions and is also a very good dataset that works well with classification models. The dataset shows transactions that were tagged as fraud and those that were legit. It is a large dataset containing 5,420,481 rows and 27 columns sourced from kaggle website (with_aggregated): https://www.kaggle.com/datasets/chendoytshman/fraud-detection-paysim

### SUMMARY OF THE DATASET
| Property | Details |
|---|---|
| **Source** | [Kaggle — PaySim Fraud Detection (with aggregated)](https://www.kaggle.com/datasets/chendoytshman/fraud-detection-paysim) |
| **Records** | 5,420,481 transactions |
| **Original Features** | 27 columns |
| **Final Features Used** | 6 selected features |
| **Target Variable** | `fraud_label` (0 = Non-Fraud, 1 = Fraud) |
| **Fraud Rate** | 1.12% (60,666 fraud / 5,359,815 non-fraud) |
| **Memory (Original)** | 1,116 MB |
| **Memory (Optimised)** | 475 MB (-57.4%) |

### FEATURES AND INTERPRETATION
## 📋 Dataset Column Description

| # | Features | Data Type | Description |
|---|---|---|---|
| 1 | `row_id` | int32 | Unique identifier for each transaction row in the dataset |
| 2 | `hour_of_simulation` | int16 | The hour within the 720-hour (30-day) simulation period in which the transaction occurred |
| 3 | `transaction_type` | category | The method of transaction — CASH_IN, CASH_OUT, DEBIT, PAYMENT or TRANSFER |
| 4 | `transaction_amount` | float32 | The total monetary value of the transaction carried out |
| 5 | `sender_id` | object | Unique identifier of the account initiating the transaction |
| 6 | `sender_balance_before` | float32 | The sender's account balance immediately before the transaction was made |
| 7 | `sender_balance_after` | float32 | The sender's account balance immediately after the transaction was completed |
| 8 | `receiver_id` | object | Unique identifier of the account receiving the transaction |
| 9 | `receiver_balance_before` | float32 | The receiver's account balance immediately before the transaction was received |
| 10 | `receiver_balance_after` | float32 | The receiver's account balance immediately after the transaction was received |
| 11 | `fraud_label` | int8 | Target variable — 0 = Legitimate transaction, 1 = Fraudulent transaction |
| 12 | `unauthorized_overdraft_flag` | int8 | Binary flag — 1 = transaction triggered an unauthorized overdraft, 0 = it did not |
| 13 | `total_sent_last_1hr` | float32 | Total cumulative amount sent by the sender in the last 1 hour |
| 14 | `total_sent_last_1day` | float32 | Total cumulative amount sent by the sender in the last 24 hours |
| 15 | `total_sent_last_7days` | float32 | Total cumulative amount sent by the sender in the last 7 days |
| 16 | `total_sent_last_30days` | float32 | Total cumulative amount sent by the sender in the last 30 days |
| 17 | `tx_count_last_1hr` | int8 | Number of transactions made by the sender in the last 1 hour |
| 18 | `tx_count_last_1day` | int16 | Number of transactions made by the sender in the last 24 hours |
| 19 | `tx_count_last_7days` | int16 | Number of transactions made by the sender in the last 7 days |
| 20 | `tx_count_last_30days` | int16 | Number of transactions made by the sender in the last 30 days |
| 21 | `avg_amount_last_1hr` | float32 | Average transaction amount sent by the sender in the last 1 hour |
| 22 | `avg_amount_last_1day` | float32 | Average transaction amount sent by the sender in the last 24 hours |
| 23 | `avg_amount_last_7days` | float32 | Average transaction amount sent by the sender in the last 7 days |
| 24 | `avg_amount_last_30days` | float32 | Average transaction amount sent by the sender in the last 30 days |
| 25 | `transaction_type_encoded` | int8 | Numerically encoded version of transaction_type — CASH_IN=0, CASH_OUT=1, DEBIT=2, PAYMENT=3, TRANSFER=4 |

Note: Three columns were dropped during cleaning and are not included above — transaction_time_duplicate, time_merge_flag and rule_based_fraud_flag — as they were identified as redundant and irrelevant to the fraud prediction objective.
















