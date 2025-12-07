# ML/DL in Disease Severity Detection: Myocardial Infarction Complications

This repository contains the code for my MSc Artificial Intelligence project on  
**disease severity detection** using machine learning and deep learning models,  
applied to the **Myocardial Infarction Complications** dataset from the UCI
Machine Learning Repository (ID: 579).

The goal is to build predictive models that estimate the risk/severity of
complications following myocardial infarction and map prediction probabilities
into clinically interpretable **risk categories** (Low, Moderate, High).

---

## 1. Dataset

- **Source:** UCI Machine Learning Repository – Myocardial Infarction Complications (ID: 579)  
- **Loading:** The dataset is fetched programmatically using `ucimlrepo.fetch_ucirepo(id=579)`.
- **Task:** Binary classification (complication vs. no complication / severity surrogate).

Preprocessing steps:

- Missing values in features are imputed with the **median**.
- Categorical variables are encoded using **LabelEncoder**.
- Numerical features are standardized using **StandardScaler**.
- Class imbalance is handled using **SMOTE** (Synthetic Minority Over-sampling Technique).

---

## 2. Methods and Models

The project implements a full supervised learning pipeline with:

### 2.1 Feature Selection

- **LASSO (L1-regularized linear model)** is used to select informative features.
- Only features with non-zero coefficients are retained for model training.

### 2.2 Models Implemented

The following models are trained and evaluated:

- **Random Forest**
- **Support Vector Machine (SVM)** with probability estimates
- **Artificial Neural Network (ANN)** (Keras / TensorFlow)
- **Naive Bayes (GaussianNB)**
- **Decision Tree**
- **K-Nearest Neighbours (KNN)**
- **XGBoost (XGBClassifier)**

### 2.3 Severity Scoring

Model output probabilities are transformed into **risk categories**:

- `prob < 0.3` → **Low Risk**
- `0.3 ≤ prob < 0.7` → **Moderate Risk**
- `prob ≥ 0.7` → **High Risk**

These thresholds can be adjusted depending on clinical requirements.

---

## 3. Validation and Evaluation

The project includes extensive evaluation:

- **Standard metrics:**
  - Accuracy
  - Precision
  - Sensitivity (Recall)
  - Specificity
  - Classification report
- **ROC & AUC:**
  - ROC curves and AUC scores for:
    - Random Forest
    - SVM
    - ANN
- **Y-Randomization (Y-scrambling):**
  - Implemented for:
    - Random Forest
    - SVM
    - ANN
  - Labels are shuffled multiple times to verify that the model does not learn
    spurious associations.
- **Bootstrap + Paired Student’s t-test:**
  - Bootstrapped AUC distributions are generated for:
    - Random Forest
    - SVM
    - ANN
  - Paired t-tests compare AUC distributions to check whether performance
    differences are statistically significant.
- **Visualization:**
  - ROC curves
  - KDE plots of bootstrapped AUCs
  - Feature importance (Random Forest)
  - Confusion matrices (RF, SVM, ANN)
  - Learning curves:
    - Sklearn `learning_curve` for RF and SVM
    - Epoch-wise training/validation curves for ANN

---

## 4. Repository Structure

A simple suggested structure:

```text
.
├─ src/
│  └─ <your_main_script>.py     # main code containing the full pipeline
├─ docs/
│  └─ Paper.pdf                 # MSc project paper (optional, if you add it)
├─ requirements.txt
└─ README.md
