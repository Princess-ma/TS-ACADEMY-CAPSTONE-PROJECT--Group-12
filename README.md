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
| Princess Chiamaka Emenari | princessemenari2@gmail.com | [Github](https://github.com/Princess-ma) | Group Leader |

---
### Active Members

| Name | Email | GitHub | Role |
|---|---|---|---|
| Kolawole Julius Oluwatobi | anthonyjk1204@gmail.com | [GitHub](https://github.com/kjuls) | Active Member |
| Olaleru Praise Ajibola | zionpraiz9@gmail.com | [GitHub](https://github.com/zionpraiz9-code) | Active Member |
| Adeleye Adekunle Oluwaseun | dequnle7@gmail.com | [GitHub](https://github.com/qunlecrown) | Active Member |
| Udoh Edidiong Monday | beeeddy22@gmail.com | [GitHub](https://github.com/Edidiong-Udoh2) | Active Member |
| Ukonu Fortune Chiemela | ukonufortune@gmail.com | [GitHub](https://github.com/Fortuneukonu) | Active Member |
| Ogunniyi Ibrahim Adedeji | ogunniyiibrahim2029@gmail.com | [GitHub](https://github.com/nobleXibrahim) | Active Member |
| Titus Oluwafemi Ojo | femititus@gmail.com | [GitHub](https://github.com/femititus) | Active Member |


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
The dataset being used is a fraud detection dataset of users who carried out transactions using paysim. PaySim is a financial simulator that simulates mobile money transactions based on an original dataset. Although the dataset was generated synthetically using paysim, this dataset was chosen because of how good it relates with real world financial transactions and is also a very good dataset that works well with classification models. The dataset shows transactions that were tagged as fraud and those that were legit. It is a large dataset containing 5,420,481 rows and 27 columns sourced from kaggle website (with_aggregated): [Kaggel Website](https://www.kaggle.com/datasets/chendoytshman/fraud-detection-paysim) 

### SUMMARY OF THE DATASET
| Property | Details |
|---|---|
| **Source** | [Kaggle — Dataset](https://www.kaggle.com/datasets/chendoytshman/fraud-detection-paysim) |
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

## 🔬 Methodology

### Stage 1 — Data Cleaning & Preparation

The project began with loading the dataset containing 5,420,481 transaction records across 27 columns. The first step was memory optimization which is a custom heuristic downcasting function and it was applied that converted 64-bit data types to their smallest valid equivalents (int8, int16, float32), reducing the dataset memory footprint by 57.4% from 1,116 MB down to 475 MB, making the 5.4 million record dataset computationally manageable for all downstream analysis. All 27 columns were renamed from cryptic original names to semantically descriptive labels — for example, `step` became `hour_of_simulation`, `action` became `transaction_type` and `nameOrig` became `sender_id`, establishing the semantic clarity that carried through every subsequent stage. Three redundant columns (`transaction_time_duplicate`, `time_merge_flag`, `rule_based_fraud_flag`) were dropped, and the only categorical text column `transaction_type` was label encoded into numerical values (CASH_IN=0, CASH_OUT=1, DEBIT=2, PAYMENT=3, TRANSFER=4). The dataset was confirmed to contain zero missing values and zero duplicate records across all 5.4 million rows, establishing it as a clean and structurally sound foundation for analysis.

### Stage 2 - Data Distribution

With the dataset cleaned, Stage 2 examined the distribution of every feature across both categorical and numerical dimensions. The most critical discovery was the severe class imbalance in the fraud label (target variable) where only 1.12% of transactions (60,666 records) are fraudulent while 98.88% (5,359,815 records) are legitimate. The transaction type analysis revealed that CASH_IN dominates the dataset at 41.6% of all transactions, followed by PAYMENT at 27.57%, CASH_OUT at 20.74%, TRANSFER at 6.73% and DEBIT at 3.31%. The hour_of_simulation column confirmed 720 unique values representing each hour of the 30-day simulation period and when grouped into weekly periods, Week 4 (Day 22–30) showed the highest transaction volume at 28.6%, foreshadowing the end-of-month fraud escalation discovered in Stage 3. Numerical analysis revealed that transaction_amount and all four rolling average amount columns are severely right-skewed and leptokurtic where transaction_amount alone has a skewness of 12.622 and kurtosis of 230.642 confirming that the majority of transactions are small while a small number of extremely large transactions create heavy right tails, a distribution shape that reflects real-world fraud behavior.

### Stage 3 - Bivariate and Multivariate Analysis

Stage 3 produced the most decisive analytical discoveries of the entire project, each one directly shaping the feature selection and model design decisions in Stage 4. The countplot of fraud by transaction type revealed that fraud is exclusively concentrated in TRANSFER (8.3% fraud rate) and CASH_OUT (2.7% fraud rate), while CASH_IN, PAYMENT and DEBIT show a 0.0% fraud rate indicating a classic Drain-and-Exit fraud workflow where funds are moved via TRANSFER then extracted via CASH_OUT. The violin and KDE plots confirmed that the average fraudulent transaction amount (2,757,426) is 22.5 times larger than the average legitimate transaction amount (122,719), with this gap persisting consistently across all four rolling time windows (1hr, 1day, 7days, 30days). The scatter plot of sender_balance_before against transaction_amount revealed the most striking fraud signature in the dataset, fraudulent transactions form a near-perfect diagonal line, meaning fraudsters systematically send amounts proportional to their entire available balance regardless of account size, a structured account-draining behavior that distinguishes them from the loosely clustered legitimate transactions near the origin. The mule account analysis identified 4,937 receiver accounts that received more than one fraudulent transfer, with the most active used 19 times and this shows an organised fraud network rather than isolated incidents. The correlation heatmap revealed severe multicollinearity among the rolling average features (0.93–0.96 with each other) and identified `avg_amount_last_30days` (corr=0.70) and `transaction_amount` (corr=0.60) as the strongest linear fraud predictors. Finally, the weekly fraud distribution confirmed a statistically significant end-of-month escalation with Week 4 reaching 29.78% fraud concentration compared to 22–24% in Weeks 1–3.

### Stage 4 — Data Preprocessing

Building directly on the evidence from Stage 3, Stage 4 translated all analytical findings into a precise, model-ready dataset. Six features were selected which are:`transaction_amount`, `avg_amount_last_30days`, `transaction_type_encoded`, `total_sent_last_1hr`, `receiver_balance_after` and `week_group_encoded` based on correlation strength, absence of multicollinearity and domain-level interpretability. Features with severe redundancy were excluded, including the three other avg_amount columns (0.93–0.96 correlation with avg_amount_last_30days), sender_balance_before (0.94 correlation with sender_balance_after) and row_id/hour_of_simulation (perfect 1.00 correlation with each other). The dataset was split 80/20 using stratified sampling with random_state=42, preserving the 1.12% fraud rate in both partitions whch produced a training set of 4,336,384 and a test set of 1,084,097. Random undersampling was applied exclusively to the training set, randomly reducing the 4,287,851 non-fraud records to match the 48,533 fraud records, producing a perfectly balanced training set of 97,066 samples. The test set was deliberately left at its original imbalanced distribution to accurately simulate real-world deployment conditions. StandardScaler was fitted exclusively on the balanced training set and used to transform both the training and test sets, ensuring mean=0 and std=1 across all 6 features without allowing the 98.88% non-fraud majority to distort the scaling parameters.

### Stage 5- Machine Learning

Two classification models were built and evaluated in this project which are Logistic Regression (baseline) and Random Forest classification (advanced) alongside a Grid Search CV hyperparameter optimization, all trained on the balanced undersampled training set of 97,066 samples from Stage 4.

The **Logistic Regression** model was configured with strong L2 regularization (C=0.05) to prevent the large-magnitude features particularly `avg_amount_last_30days` and `transaction_amount` from dominating the decision boundary. It achieved a train accuracy of 95.80%, test accuracy of 98.98% and test recall of 92.58%, correctly catching 11,233 out of 12,133 fraud cases. The test precision of 52.49% and F1 of 67.00% reflect the expected challenge of a linear model applied to a non-linear, imbalanced fraud problem. Feature coefficients confirmed every Stage 3 finding — `avg_amount_last_30days` led at +10.293, `transaction_amount` followed at +7.228, and `week_group_encoded` at +0.680 validated the end-of-month temporal signal despite its near-zero Pearson correlation.
### Logistic Regression
| Metric | Train | Test |
|---|---|---|
| Accuracy | 95.80% | 98.98% |
| Precision | 99.02% | 52.49% |
| Recall | 92.51% | 92.58% |
| F1 Score | 95.66% | 67.00% |

**Confusion Matrix — Test Set**
|  | Predicted Non-Fraud | Predicted Fraud |
|---|---|---|
| **Actual Non-Fraud** | 1,061,798 | 10,166 |
| **Actual Fraud** | 900 | 11,233 |

The **Random Forest** outperformed Logistic Regression across every fraud-specific metric. Configured with n_estimators=100, max_depth=8 and min_samples_leaf=50 to prevent overfitting, it achieved a test recall of 97.50% and F1 of 71.46% reducing false negatives from 900 to just 303 and false positives from 10,166 to 9,146 compared to Logistic Regression. Feature importance by Gini impurity ranked `transaction_amount` first at 42.54%, `avg_amount_last_30days` second at 26.39%, `total_sent_last_1hr` third at 17.55%, `transaction_type_encoded` fourth at 9.95%, `receiver_balance_after` fifth at 3.23% and `week_group_encoded` last at 0.34% validating all feature selection decisions from Stage 4 and confirming that the behavioral fraud patterns discovered in Stages 2 and 3 translated directly into model-level predictive power.
### Random Forest
| Metric | Train | Test |
|---|---|---|
| Accuracy | 98.40% | 98.98% |
| Precision | 99.22% | 56.40% |
| Recall | 97.56% | 97.50% |
| F1 Score | 98.38% | 71.46% |

**Confusion Matrix — Test Set**
|  | Predicted Non-Fraud | Predicted Fraud |
|---|---|---|
| **Actual Non-Fraud** | 1,062,818 | 9,146 |
| **Actual Fraud** | 303 | 11,830 |

The **Grid Search CV** evaluated 162 parameter combinations across 486 model fits using ROC-AUC as the scoring metric, identifying the optimal configuration as max_depth=10, min_samples_leaf=25, min_samples_split=50, n_estimators=100 and max_features='sqrt'. This achieved a best cross-validation ROC-AUC of 0.9993, improving fraud precision to 0.60, recall to 0.98 and F1 to 0.75 which confirms that the baseline (random forest) parameters were near-optimal while delivering meaningful operational improvements that make the Grid Search optimised Random Forest the recommended production model.
### Grid Search CV — Optimised Random Forest
```
Best Parameters:
  max_depth         : 10
  max_features      : 'sqrt'
  min_samples_leaf  : 25
  min_samples_split : 50
  n_estimators      : 100

Best CV ROC-AUC    : 0.9993 (99.93%)
```
| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| Non-Fraud | 1.00 | 0.99 | 1.00 |
| **Fraud** | **0.60** | **0.98** | **0.75** |
| Overall Accuracy | | | **99%** |

> **Best Model: Grid Search optimised Random Forest**
> — Catches 98 in every 100 fraudulent transactions
> — Only 303 missed fraud cases out of 12,133 total

## RESULT SUMMARY
```
╔══════════════════════════════════════════════════════════╗
║           FRAUD DETECTION — FINAL RESULTS               ║
╠══════════════════════════════════════════════════════════╣
║  Dataset          : Dataset (5,420,481 transactions)     ║
║  Fraud Rate       : 1.12%                               ║
║  Features Used    : 6                                   ║
║  Best Model       : Random Forest (Grid Search Tuned)   ║
║                                                          ║
║  Test Recall      : 97.50% → 98.00% (after GS)         ║
║  Test Precision   : 56.40% → 60.00% (after GS)         ║
║  Test F1 Score    : 71.46% → 75.00% (after GS)         ║
║  ROC-AUC          : 0.9993                              ║
║                                                          ║
║  Fraud Cases Caught    : 11,830 / 12,133 (97.5%)        ║
║  Fraud Cases Missed    : 303 / 12,133   (2.5%)          ║
╚══════════════════════════════════════════════════════════╝
```

---
## Conclusion and Recommendations
### Recommendations

Based on the findings of the analysis, the organisation should strengthen its fraud prevention framework by focusing on the most significant risk indicators identified in the dataset. 
- First, real-time velocity monitoring should be implemented so that transactions associated with unusually high values of total_sent_last_1hr, particularly those above the 75th percentile threshold of 304,943, trigger immediate secondary authentication. This is necessary because rapid spending within a short period emerged as the strongest short-term fraud signal.
- In addition, TRANSFER and CASH_OUT transactions should be subjected to stricter verification procedures. Given their fraud rates of 8.3% and 2.7% respectively, these transaction types present greater exposure to fraudulent activity, especially where the transaction amount is close to the sender’s available balance. Measures such as one-time passwords or biometric confirmation would help reduce this risk.
- The organisation should also adopt end-of-month fraud response measures. Since Week 4 recorded the highest fraud concentration at 29.78%, compared with 22–24% in earlier weeks, fraud monitoring systems should be more sensitive and investigation teams better prepared during the final seven days of each month.
- Furthermore, repeat receiver accounts linked to fraud should be continuously monitored. The 4,937 mule accounts identified in Stage 3 should be flagged in real time, and any transaction involving them should automatically prompt a fraud review.
- Finally, future versions of the fraud detection model should include additional variables such as sender account age, device fingerprint consistency, and geographic velocity. These features would improve the model’s ability to detect more complex fraud patterns not fully captured by the current six-feature model.

### CONCLUSION

The Capstone Project carried out by the prestigious members of Group 12 project demonstrates a comprehensive fraud detection workflow that includes data preparation, exploratory analysis, visualization, and machine learning model optimization. The analysis revealed key behavioral patterns within transaction data and highlighted the challenge of detecting fraud within highly imbalanced datasets. Exploratory analysis revealed that fraudulent activities occur primarily in TRANSFER and CASH-OUT transactions, and that fraud cases are extremely rare compared to normal transactions. Visualization techniques further helped identify patterns and anomalies within the data. Through hyperparameter tuning, the Random Forest model achieved improved performance, demonstrating its effectiveness in identifying potentially fraudulent transactions within the dataset.
These findings highlights the value of combining data analysis, visualization and machine learning to strengthen fraud detection systems in financial institutions. Such systems are essential for improving security in digital financial services and reducing the risk of financial fraud.

Overall, the results show that machine learning models play an important role in improving fraud detection systems and helping financial institutions identify suspicious transactions more effectively.

## ACKNOLEDGEMENTS
- Hart Ofigwe (Data Science Tutor)
- The TS Academy Scholarship Board
- The Group 12 Team for their collaboration and team spirit
- Kaggle for the Fraud Detection-Paysim Dataset
- Scikit-learn team for developing machine learning libraries
- GitHub for the deployment platform of our Fraud Detection Project

# References
### Data Source
- Lopez-Rojas, E. A., Elmir, A., & Axelson, S. (2016). PaySim: A financial mobile money simulator for fraud detection. In 28th European Modeling and Simulation Symposium (EMSS).
- chendoytshman. (2023). Fraud Detection - PaySim (with aggregated) [Data set]. Kaggle. https://www.kaggle.com/datasets/chendoytshman/fraud-detection-paysim
### Software & Libraries
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,

---

**License**: Apache License 2.0
**Copyright**: © 2026 TS Academy Capstone Project – Group 12

















