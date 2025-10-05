# CTG Fetal Distress Classification

**MLDA@EEE Datathon 2025 - Lifeline Challenge**

A multi-model machine learning pipeline for detecting fetal distress from cardiotocography (CTG) data, addressing severe class imbalance through SMOTE oversampling and optimizing for balanced accuracy and macro F1-score.

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Results Summary](#results-summary)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Dataset Information](#dataset-information)
8. [Methodology](#methodology)
9. [Model Architecture](#model-architecture)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Key Features & Clinical Interpretation](#key-features--clinical-interpretation)
12. [Files Generated](#files-generated)
13. [Submission Deliverables](#submission-deliverables)
14. [Technical Details](#technical-details)
15. [Troubleshooting](#troubleshooting)
16. [Acknowledgments](#acknowledgments)
17. [License](#license)

---

## 1. Overview

Cardiotocography (CTG) monitors fetal heart rate and uterine contractions during labor. This project uses machine learning to automatically classify CTG recordings as:

- **Normal (Class 1)**: Baby is healthy, no immediate concern  
- **Suspect (Class 2)**: Warning signs present, needs monitoring  
- **Pathologic (Class 3)**: High risk of distress, urgent attention required  

**The challenge:** Only 8% of cases are pathologic, creating severe class imbalance that standard ML approaches fail to handle.

---

## 2. Problem Statement

**How can we build a solution that interprets patterns in a baby's heart rate and mother's contractions to automatically identify signs of fetal distress that might otherwise go unnoticed during labor?**

Real hospital wards are busy and understaffed. Subtle warning signs like delayed heart rate recovery after contractions can be missed.  
This system acts as a "second pair of eyes" to catch critical patterns.

---

## 3. Results Summary

### Best Performing Model: Gradient Boosting

| Metric | Score |
|--------|-------|
| Balanced Accuracy | 85.99% |
| Macro F1-Score | 87.75% |
| Weighted F1-Score | 93.79% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| Normal | 95.67% | 98.41% | 97.02% | 314 |
| Suspect | 86.36% | 71.70% | 78.35% | 53 |
| **Pathologic** | **87.88%** | **87.88%** | **87.88%** | **33** |

**Critical Clinical Impact:**  
87.88% recall on pathologic cases means the model correctly identifies nearly 9 out of 10 dangerous situations requiring immediate intervention.  
Only 2 pathologic cases were misclassified as Normal.

---

## 4. Repository Structure

ctg-fetal-distress-classification/
│
├── README.md # This file
├── requirements.txt # Python dependencies
├── train.py # Training script
├── test.py # Inference/testing script
├── ctg_fetal_distress_classification.ipynb # Complete Jupyter notebook
│
├── data/
│ ├── CTG.xls # Raw UCI CTG dataset
│ └── ctg_cleaned.csv # Preprocessed data
│
├── results/
│ ├── cv_results.csv # Cross-validation scores
│ ├── test_results.csv # Final test performance
│ ├── feature_importance.csv # Feature rankings
│ └── visualizations/
│ ├── Logistic_Regression_confusion_matrix.png
│ ├── Random_Forest_confusion_matrix.png
│ ├── Gradient_Boosting_confusion_matrix.png
│ ├── MLP_Neural_Network_confusion_matrix.png
│ ├── feature_importance.png
│ ├── model_comparison.png
│ └── per_class_recall.png
│
└── docs/academic_report.pdf # Technical report
└── demo/demo_video.mp4

## 5. Installation

### Prerequisites
- Python 3.7 or higher  
- pip package manager

### Step 1: Clone Repository
git clone https://github.com/yourusername/ctg-fetal-distress-classification.git
cd ctg-fetal-distress-classification
Step 2: Install Dependencies
pip install -r requirements.txt

Dependencies installed:
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
imbalanced-learn >= 0.9.0
shap >= 0.40.0
joblib

6. Usage
Option 1: Run Complete Notebook (Recommended for Exploration)
Open ctg_fetal_distress_classification.ipynb in:

Google Colab: Upload the notebook

Jupyter Notebook:
jupyter notebook ctg_fetal_distress_classification.ipynb

Run all cells sequentially. The notebook includes:
1. Data cleaning classes
2. Exploratory data analysis
3. Model training
4. Evaluation and Visualizations

Option 2: Run Training Script
python train.py

What happens:
1. Loads data/ctg_cleaned.csv
2. Performs 80/20 stratified train/test split
3. Trains 4 models with hyperparameter tuning (15–20 iterations each)
4. Uses 5-fold cross-validation
5. Saves trained models to models/ directory
Expected runtime: 10–15 minutes on standard hardware

Output:
Console: Training progress, CV scores, best parameters
Files: models/*.pkl files saved

Option 3: Run Testing/Inference Script
python test.py

Prerequisites: Must run train.py first to generate model files.
What happens:
1. Loads trained models from models/ directory
2. Loads test data (same split as training)
3. Evaluates each model on test set
4. Generates confusion matrices
5. Saves results to CSV

Output:
Console: Test metrics for all models
Files: test_results.csv, confusion matrix PNGs

7. Dataset Information
Source
UCI Machine Learning Repository - Cardiotocography Dataset

Description
2,126 total samples (CTG recordings)
35 features (numeric only)
3 classes (Normal / Suspect / Pathologic)
Labeled by consensus of 3 expert obstetricians

Class Distribution (Severe Imbalance)
Class	Count	Percentage
Normal (1)	1655	77.8%
Suspect (2)	295	13.9%
Pathologic (3)	176	8.3%

Why this matters:
A naive model predicting “Normal” for everything would achieve 78% accuracy but miss ALL critical cases.
Hence, we use balanced metrics.

Key Features
Baseline & Variability
LB: Baseline fetal heart rate (bpm)
ASTV: Abnormal short-term variability
ALTV: Abnormal long-term variability
mSTV: Mean short-term variability
mLTV: Mean long-term variability
Accelerations & Movements
AC: Number of accelerations
FM: Number of fetal movements
Decelerations (Critical Danger Signs)
DL: Light decelerations
DS: Severe decelerations
DP: Prolonged decelerations
DR: Repetitive decelerations

Uterine Contractions
UC: Number of uterine contractions

Histogram Features
Width, Min, Max, Mode, Mean, Median, Variance, Tendency

8. Methodology
Three-Strategy Approach
1. SMOTE Oversampling
a) Generates synthetic minority class samples (Suspect, Pathologic)
b) Implementation:
SMOTE(random_state=42, k_neighbors=5)
Tested k_neighbors values: [3, 5, 7]

2. Stratified Cross-Validation
a) Preserves class proportions in every fold
b) Implementation:
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

3. Balanced Evaluation Metrics
a) Optimizes for Macro F1-Score instead of accuracy
b) Parameter:
scoring='f1_macro'


9. Model Architecture
Pipeline Structure
Input Data → StandardScaler → SMOTE → Classifier → Predictions

Stage 1: StandardScaler
Normalizes features (mean=0, std=1)

Prevents large-scale features from dominating

Stage 2: SMOTE
Balances training data by generating synthetic minority samples

Stage 3: Classifier
Trains and predicts using chosen ML algorithm

Models Implemented
1. Logistic Regression
Purpose: Interpretable baseline

Configuration:
Multinomial solver: L-BFGS
L2 regularization
Class weights: balanced
Max iterations: 4000
Tuned Parameters:
C (regularization strength), k_neighbors (SMOTE)

2. Random Forest
Purpose: Handles feature interactions robustly

Configuration:
500–1200 trees
Class weights: balanced_subsample
Bootstrap: enabled
Tuned Parameters:
n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features

3. Gradient Boosting (Best Performer)
Purpose: Sequential learning, focuses on hard cases

Configuration:
BorderlineSMOTE
Subsample: 0.8
200–500 estimators
Tuned Parameters:
n_estimators, learning_rate, max_depth, min_samples_split, subsample

4. MLP Neural Network
Purpose: Captures complex non-linear patterns

Configuration:
Hidden layers: (128,64), (256,128,64), (128,64,32), (256,128)
Activation: ReLU
Early stopping: enabled
Tuned Parameters:
hidden_layer_sizes, alpha, learning_rate_init, batch_size

10. Evaluation Metrics
Why Not Accuracy?
With 78% Normal cases, a model predicting “Normal” for everything achieves 78% accuracy but has 0% recall for pathologic cases. This is clinically catastrophic.

Metrics Used
1. Balanced Accuracy
Average recall across all classes
(Recall_Normal + Recall_Suspect + Recall_Pathologic) / 3

2. Macro F1-Score
Unweighted average of per-class F1 scores
(F1_Normal + F1_Suspect + F1_Pathologic) / 3

3. Per-Class Metrics
Precision: Correct predictions / Predicted positives
Recall: Correct predictions / Actual positives
F1-Score: Harmonic mean of precision & recall

Most Critical Metric for Clinical Impact:
Pathologic Recall = 87.88% → 29 out of 33 dangerous cases correctly identified.

11. Key Features & Clinical Interpretation
Top 10 Most Important Features (Permutation Importance)
Rank	Feature	Importance	Clinical Meaning
1	ASTV	0.188	Abnormal short-term variability
2	ALTV	0.180	Abnormal long-term variability
3	mSTV	0.217	Mean short-term variability
4	AC	0.134	Accelerations (healthy sign)
5	Median	0.153	Median heart rate
6	Mean	0.162	Mean heart rate
7	AC.1	0.138	Alternate acceleration measure
8	DP	0.0158	Prolonged decelerations
9	DP.1	0.0129	Alternate prolonged deceleration measure
10	Min	0.0124	Minimum heart rate recorded

Medical Validation:
Low ASTV + high DP → Danger (oxygen deprivation)
High AC + normal baseline → Reassuring
Multiple deceleration types → Concerning pattern
This alignment with medical knowledge confirms model interpretability.

12. Files Generated
1. During Training (train.py)
models/
a) Logistic_Regression_model.pkl
b) Random_Forest_model.pkl
c) Gradient_Boosting_model.pkl
d) MLP_Neural_Network_model.pkl

2. During Testing (test.py)
a) test_results.csv
b) Confusion matrices (.png)

3. From Complete Notebook
a) cv_results.csv
b) feature_importance.csv
c) feature_importance.png
d) model_comparison.png
e) per_class_recall.png


13. Technical Details
Hyperparameter Tuning
Method: RandomizedSearchCV

Combinations: 15–20 random sets per model

Cross-validation: 5-fold stratified

Optimization metric: Macro F1

Why RandomizedSearchCV?
Efficient, explores broader search space faster.

Cross-Validation Strategy
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 - Maintains 78%/13%/8% distribution per fold.

Data Splitting
 - train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
 - 80% train / 20% test
 - Stratified for class balance
 - Fixed seed for reproducibility

Model Saving
joblib.dump(estimator, 'models/model_name.pkl')

Loading:
model = joblib.load('models/model_name.pkl')


14. Acknowledgments
Dataset: UCI Machine Learning Repository
Competition: MLDA@EEE Datathon 2025
Theme: Lifeline – Fetal Distress Detection


