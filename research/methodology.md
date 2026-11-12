
# Research Methodology

## 1. Dataset Description

The CIC-IDS2017 dataset was selected as the primary dataset for network traffic anomaly detection.

The dataset consists of eight CSV files containing a total of 2,830,743 network traffic records.

Each file contains 79 columns:
- 78 input features.
- 1 target column (`Label`).

All eight CSV files were verified to have identical column structures.

## 2. Class Distribution Analysis

The dataset contains 15 traffic classes, including benign traffic and multiple attack categories.

The class distribution analysis identified a significant imbalance:

- BENIGN: 2,273,097 records (80.30%).
- DoS Hulk: 231,073 records (8.16%).
- PortScan: 158,930 records (5.61%).
- DDoS: 128,027 records (4.52%).

Several attack categories, including Infiltration, SQL Injection and Heartbleed, contain very few records.

Therefore, model evaluation will consider precision, recall, macro F1-score, PR-AUC and false-positive rate rather than relying exclusively on accuracy.

## 3. Dataset Quality Assessment

Initial data-quality analysis identified:

- 1,358 missing values.
- 4,376 infinite values.
- Two affected features: `Flow Packets/s` and `Flow Bytes/s`.

Further investigation established that all 2,867 records containing invalid flow-rate values also had a flow duration of zero.

These observations suggest that division-by-zero or undefined rate calculations are responsible for the invalid flow-rate values.

## 4. Proposed Data Preprocessing Strategy

The original dataset files will remain unchanged.

The preprocessing pipeline will:

1. Convert positive and negative infinity values into missing values.
2. Preserve the original `Flow Duration` feature.
3. Impute missing numerical values using median values calculated exclusively from the training dataset.
4. Apply the same learned imputation parameters to validation and test datasets.
5. Investigate duplicate records before deciding how to handle them.
6. Encode target labels appropriately for supervised learning.

A sensitivity experiment will compare median imputation against the removal of records containing invalid flow-rate values.

## 5. Prevention of Data Leakage

All preprocessing parameters, including imputation values and scaling parameters, will be learned using training data only.

Validation and test datasets will not influence preprocessing decisions or model training.

Dataset splitting will consider temporal and attack-scenario separation to reduce the risk of overly optimistic evaluation results.

## 6. Model Evaluation

The proposed models will be evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- ROC-AUC
- PR-AUC
- False-positive rate
- Detection latency
- Throughput

The final evaluation will compare supervised, unsupervised and hybrid anomaly detection approaches.
