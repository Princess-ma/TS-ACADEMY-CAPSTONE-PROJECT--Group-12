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

## 🔬 Methodology

### Stage 1 — Data Cleaning & Preparation

The project began with loading the dataset containing 5,420,481 transaction records across 27 columns. The first step was memory optimization which is a custom heuristic downcasting function and it was applied that converted 64-bit data types to their smallest valid equivalents (int8, int16, float32), reducing the dataset memory footprint by 57.4% from 1,116 MB down to 475 MB, making the 5.4 million record dataset computationally manageable for all downstream analysis. All 27 columns were renamed from cryptic original names to semantically descriptive labels — for example, `step` became `hour_of_simulation`, `action` became `transaction_type` and `nameOrig` became `sender_id`, establishing the semantic clarity that carried through every subsequent stage. Three redundant columns (`transaction_time_duplicate`, `time_merge_flag`, `rule_based_fraud_flag`) were dropped, and the only categorical text column `transaction_type` was label encoded into numerical values (CASH_IN=0, CASH_OUT=1, DEBIT=2, PAYMENT=3, TRANSFER=4). The dataset was confirmed to contain zero missing values and zero duplicate records across all 5.4 million rows, establishing it as a clean and structurally sound foundation for analysis.

### Stage 2 - Data Distribution

With the dataset cleaned, Stage 2 examined the distribution of every feature across both categorical and numerical dimensions. The most critical discovery was the severe class imbalance in the fraud label (target variable) where only 1.12% of transactions (60,666 records) are fraudulent while 98.88% (5,359,815 records) are legitimate. The transaction type analysis revealed that CASH_IN dominates the dataset at 41.6% of all transactions, followed by PAYMENT at 27.57%, CASH_OUT at 20.74%, TRANSFER at 6.73% and DEBIT at 3.31%. The hour_of_simulation column confirmed 720 unique values representing each hour of the 30-day simulation period and when grouped into weekly periods, Week 4 (Day 22–30) showed the highest transaction volume at 28.6%, foreshadowing the end-of-month fraud escalation discovered in Stage 3. Numerical analysis revealed that transaction_amount and all four rolling average amount columns are severely right-skewed and leptokurtic where transaction_amount alone has a skewness of 12.622 and kurtosis of 230.642 confirming that the majority of transactions are small while a small number of extremely large transactions create heavy right tails, a distribution shape that reflects real-world fraud behavior.

### Stage 3 - Bivariate and Multivariate Analysis

Stage 3 produced the most decisive analytical discoveries of the entire project, each one directly shaping the feature selection and model design decisions in Stage 4. The countplot of fraud by transaction type revealed that fraud is exclusively concentrated in TRANSFER (8.3% fraud rate) and CASH_OUT (2.7% fraud rate), while CASH_IN, PAYMENT and DEBIT show a 0.0% fraud rate indicating a classic Drain-and-Exit fraud workflow where funds are moved via TRANSFER then extracted via CASH_OUT. The violin and KDE plots confirmed that the average fraudulent transaction amount (2,757,426) is 22.5 times larger than the average legitimate transaction amount (122,719), with this gap persisting consistently across all four rolling time windows (1hr, 1day, 7days, 30days). The scatter plot of sender_balance_before against transaction_amount revealed the most striking fraud signature in the dataset, fraudulent transactions form a near-perfect diagonal line, meaning fraudsters systematically send amounts proportional to their entire available balance regardless of account size, a structured account-draining behavior that distinguishes them from the loosely clustered legitimate transactions near the origin. The mule account analysis identified 4,937 receiver accounts that received more than one fraudulent transfer, with the most active used 19 times and this shows an organised fraud network rather than isolated incidents. The correlation heatmap revealed severe multicollinearity among the rolling average features (0.93–0.96 with each other) and identified `avg_amount_last_30days` (corr=0.70) and `transaction_amount` (corr=0.60) as the strongest linear fraud predictors. Finally, the weekly fraud distribution confirmed a statistically significant end-of-month escalation with Week 4 reaching 29.78% fraud concentration compared to 22–24% in Weeks 1–3.

### Stage 4 — Data Preprocessing

Building directly on the evidence from Stage 3, Stage 4 translated all analytical findings into a precise, model-ready dataset. Six features were selected which are:`transaction_amount`, `avg_amount_last_30days`, `transaction_type_encoded`, `total_sent_last_1hr`, `receiver_balance_after` and `week_group_encoded` based on correlation strength, absence of multicollinearity and domain-level interpretability. Features with severe redundancy were excluded, including the three other avg_amount columns (0.93–0.96 correlation with avg_amount_last_30days), sender_balance_before (0.94 correlation with sender_balance_after) and row_id/hour_of_simulation (perfect 1.00 correlation with each other). The dataset was split 80/20 using stratified sampling with random_state=42, preserving the 1.12% fraud rate in both partitions whch produced a training set of 4,336,384 and a test set of 1,084,097. Random undersampling was applied exclusively to the training set, randomly reducing the 4,287,851 non-fraud records to match the 48,533 fraud records, producing a perfectly balanced training set of 97,066 samples. The test set was deliberately left at its original imbalanced distribution to accurately simulate real-world deployment conditions. StandardScaler was fitted exclusively on the balanced training set and used to transform both the training and test sets, ensuring mean=0 and std=1 across all 6 features without allowing the 98.88% non-fraud majority to distort the scaling parameters.


### Stage 5 — Machine Learning
Two classification models were built and evaluated — Logistic Regression and Random Forest — alongside a Grid Search CV hyperparameter optimization on the Random Forest, with all models trained on the balanced undersampled training set of 97,066 samples produced in Stage 4.

The **Logistic Regression** model was configured with strong L2 regularization (C=0.05, solver=lbfgs, max_iter=1000). The regularization was deliberately set strong because without it, the large magnitude features identified in Stage 3 — particularly `avg_amount_last_30days` and `transaction_amount` — would produce disproportionately large coefficients that overwhelm the decision boundary and cause overfitting on the balanced training distribution. The model was trained on the scaled training data (X_train_scaled) and achieved a train accuracy of 95.80% and test accuracy of 98.98%, with a train recall of 92.51% and test recall of 92.58% — confirming strong generalization on fraud detection. The test precision of 52.49% and F1 score of 67.00% reflect the expected challenge of a linear model applied to a non-linear fraud problem where the real-world test set is 98.88% non-fraud. The confusion matrix on the test set showed 11,233 true positives (correctly caught fraud), 900 false negatives (missed fraud), 1,061,798 true negatives and 10,166 false positives (legitimate transactions incorrectly flagged).

The feature importance analysis for Logistic Regression expressed through learned coefficients confirmed every key finding from the Stage 3 EDA. `avg_amount_last_30days` led with a coefficient of +10.293, making it the single strongest fraud signal — directly aligned with its 0.70 correlation from Stage 3 and the 15–20x average amount gap identified in the bivariate analysis. `transaction_amount` followed at +7.228, consistent with its 0.60 correlation and the violin plot findings. `week_group_encoded` carried a positive coefficient of +0.680, validating the decision to retain the end-of-month temporal signal despite its near-zero Pearson correlation. `transaction_type_encoded` at +0.314 confirms TRANSFER and CASH_OUT are correctly weighted toward fraud. `total_sent_last_1hr` at −0.280 and `receiver_balance_after` at −1.274 both carry negative coefficients — meaning higher values in these features lean toward non-fraud — which aligns with the Stage 3 finding that legitimate users have higher sustained transaction counts and receiver balances increase normally after legitimate transactions.

The **Random Forest** model was chosen over a single Decision Tree because the ensemble of 100 independent trees corrects the high variance problem that a single tree suffers from on a 5.4 million record dataset with extreme skewness (transaction_amount skew=12.622, kurtosis=230.642). Each tree in the forest is trained on a random bootstrap sample and considers only sqrt(6)=2 features at each split, decorrelating the individual trees and preventing any single dominant feature from controlling all splits. The model was configured with max_depth=8, min_samples_split=100 and min_samples_leaf=50 to prevent overfitting — without depth constraints, trees would grow deep enough to memorize individual training records rather than learn generalizable fraud patterns. Unlike Logistic Regression, Random Forest was trained directly on unscaled data (X_train_balanced) since tree-based splits operate on threshold values rather than distance or gradient calculations. The Random Forest achieved a train accuracy of 98.40% and test accuracy of 98.98%, with a train recall of 97.56% and test recall of 97.50% — the near-perfect consistency between train and test recall confirms that the depth constraints effectively controlled overfitting. The test F1 score of 71.46% outperforms Logistic Regression's 67.00%, and the confusion matrix showed only 303 false negatives compared to 900 for Logistic Regression — meaning the forest caught 597 more fraud cases — alongside 9,146 false positives compared to 10,166, representing 1,020 fewer false alarms simultaneously.

The feature importance analysis for Random Forest measured by Gini impurity reduction across all 100 trees produced a ranking that both validates the Stage 3 correlation findings and reveals important non-linear distinctions. `transaction_amount` ranked first at 42.54% importance — outranking `avg_amount_last_30days` (26.39%) despite having a lower Pearson correlation (0.60 vs 0.70) — because Random Forest measures non-linear Gini reduction at each split threshold rather than linear correlation, and transaction_amount provides more powerful immediate splitting boundaries across the 97,066 balanced training records. `total_sent_last_1hr` ranked third at 17.55%, higher than its relative position in Logistic Regression, confirming that short-window velocity is a non-linear fraud signal that tree-based models capture more effectively than linear models. `transaction_type_encoded` followed at 9.95%, reflecting the decisive channel-based split between TRANSFER/CASH_OUT and the three safe transaction types. `receiver_balance_after` contributed 3.23% and `week_group_encoded` contributed 0.34% — both serving as supporting signals rather than primary predictors, consistent with their roles identified throughout the EDA.

The **Grid Search CV** evaluated all 162 combinations from the parameter grid (n_estimators: [50, 100, 200], max_depth: [6, 8, 10], min_samples_split: [50, 100, 200], min_samples_leaf: [25, 50, 100], max_features: ['sqrt', 'log2']) using 3-fold cross-validation and ROC-AUC as the scoring metric — totalling 486 model fits. The best parameters identified were max_depth=10, max_features='sqrt', min_samples_leaf=25, min_samples_split=50 and n_estimators=100, achieving a best cross-validation ROC-AUC of 0.9993 (99.93%). The increase in max_depth from 8 to 10 confirmed that slightly deeper trees better capture the complex non-linear interaction patterns between transaction_amount and transaction_type identified in Stage 3. The reduction in min_samples_leaf from 50 to 25 and min_samples_split from 100 to 50 allowed finer, more granular splits that better differentiate the concentrated fraud signatures around high-amount TRANSFER and CASH_OUT transactions. The optimised model achieved a test fraud precision of 0.60, recall of 0.98 and F1 score of 0.75 — improvements of +3.60%, +0.50% and +3.54% respectively over the baseline Random Forest — confirming that the manually tuned baseline parameters were near-optimal and that the Grid Search refinements, while incremental, translate into meaningful operational improvements in a real-world fraud detection deployment.
















