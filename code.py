import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Fetch dataset
dataset = fetch_ucirepo(id=579)
X = dataset.data.features.copy()
y = dataset.data.targets.copy()

# Convert target variable `y` into a Pandas Series
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# Handle missing values
X = X.fillna(X.median())

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

# Normalize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Encode target variable if categorical
y = LabelEncoder().fit_transform(y) if y.dtypes == 'object' else y

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Feature Selection using LASSO
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
important_features = X.columns[lasso.coef_ != 0]
X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]

print(f"Selected Features using LASSO: {important_features.tolist()}")

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_selected, y_train)

# Get probability predictions for class 1
probs = rf_model.predict_proba(X_test_selected)[:, 1]

# Function to assign severity levels
def severity_score(prob):
    if prob < 0.3:
        return "Low Risk"
    elif 0.3 <= prob < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

# Convert probabilities to severity scores
severity_labels = [severity_score(p) for p in probs]

# Print some sample severity scores
for i in range(10):
    print(f"Sample {i+1}: Probability={probs[i]:.3f}, Severity={severity_labels[i]}")

# Classification Report
y_pred = (probs > 0.5).astype(int)  # Convert probabilities to binary predictions
print(classification_report(y_test, y_pred))

# Neural Network Model for Comparison
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_selected.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_selected, y_train, epochs=50, batch_size=16, validation_data=(X_test_selected, y_test))

# Get NN-based probabilities
probs_nn = model.predict(X_test_selected).flatten()

# Convert to severity levels
severity_labels_nn = [severity_score(p) for p in probs_nn]

# Print NN-based severity scores
for i in range(10):
    print(f"NN Sample {i+1}: Probability={probs_nn[i]:.3f}, Severity={severity_labels_nn[i]}")
def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # Avoid division by zero
    
    return accuracy, precision, sensitivity, specificity

# ---- Random Forest Metrics ----
# Training Predictions
y_train_pred = rf_model.predict(X_train_selected)
accuracy_train_rf, precision_train_rf, sensitivity_train_rf, specificity_train_rf = compute_metrics(y_train, y_train_pred)

# Test Predictions
y_test_pred = (probs > 0.5).astype(int)  # Convert probabilities to binary predictions
accuracy_test_rf, precision_test_rf, sensitivity_test_rf, specificity_test_rf = compute_metrics(y_test, y_test_pred)

print("Random Forest Model Metrics:")
print(f"Training Accuracy: {accuracy_train_rf:.3f}, Precision: {precision_train_rf:.3f}, Sensitivity: {sensitivity_train_rf:.3f}, Specificity: {specificity_train_rf:.3f}")
print(f"Test Accuracy: {accuracy_test_rf:.3f}, Precision: {precision_test_rf:.3f}, Sensitivity: {sensitivity_test_rf:.3f}, Specificity: {specificity_test_rf:.3f}")

# ---- Neural Network Metrics ----
y_train_pred_nn = (model.predict(X_train_selected).flatten() > 0.5).astype(int)
y_test_pred_nn = (probs_nn > 0.5).astype(int)

accuracy_train_nn, precision_train_nn, sensitivity_train_nn, specificity_train_nn = compute_metrics(y_train, y_train_pred_nn)
accuracy_test_nn, precision_test_nn, sensitivity_test_nn, specificity_test_nn = compute_metrics(y_test, y_test_pred_nn)

print("\nNeural Network Model Metrics:")
print(f"Training Accuracy: {accuracy_train_nn:.3f}, Precision: {precision_train_nn:.3f}, Sensitivity: {sensitivity_train_nn:.3f}, Specificity: {specificity_train_nn:.3f}")
print(f"Test Accuracy: {accuracy_test_nn:.3f}, Precision: {precision_test_nn:.3f}, Sensitivity: {sensitivity_test_nn:.3f}, Specificity: {specificity_test_nn:.3f}")

#SVM (Support Vector Machine) MODEL
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_selected, y_train)

# Predict probabilities and binary predictions
probs_svm = svm_model.predict_proba(X_test_selected)[:, 1]
y_test_pred_svm = (probs_svm > 0.5).astype(int)

# Severity scores
severity_labels_svm = [severity_score(p) for p in probs_svm]

# Metrics
accuracy_svm, precision_svm, sensitivity_svm, specificity_svm = compute_metrics(y_test, y_test_pred_svm)

print("\nSVM Model Metrics:")
print(f"Test Accuracy: {accuracy_svm:.3f}, Precision: {precision_svm:.3f}, Sensitivity: {sensitivity_svm:.3f}, Specificity: {specificity_svm:.3f}")


#NAIVE BAYES

nb_model = GaussianNB()
nb_model.fit(X_train_selected, y_train)

probs_nb = nb_model.predict_proba(X_test_selected)[:, 1]
y_test_pred_nb = (probs_nb > 0.5).astype(int)

severity_labels_nb = [severity_score(p) for p in probs_nb]

accuracy_nb, precision_nb, sensitivity_nb, specificity_nb = compute_metrics(y_test, y_test_pred_nb)

print("\nNaive Bayes Model Metrics:")
print(f"Test Accuracy: {accuracy_nb:.3f}, Precision: {precision_nb:.3f}, Sensitivity: {sensitivity_nb:.3f}, Specificity: {specificity_nb:.3f}")


#DECISION TREE

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_selected, y_train)

probs_dt = dt_model.predict_proba(X_test_selected)[:, 1]
y_test_pred_dt = (probs_dt > 0.5).astype(int)

severity_labels_dt = [severity_score(p) for p in probs_dt]

accuracy_dt, precision_dt, sensitivity_dt, specificity_dt = compute_metrics(y_test, y_test_pred_dt)

print("\nDecision Tree Model Metrics:")
print(f"Test Accuracy: {accuracy_dt:.3f}, Precision: {precision_dt:.3f}, Sensitivity: {sensitivity_dt:.3f}, Specificity: {specificity_dt:.3f}")


#K-Nearest Neighbour

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_selected, y_train)

probs_knn = knn_model.predict_proba(X_test_selected)[:, 1]
y_test_pred_knn = (probs_knn > 0.5).astype(int)

accuracy_knn, precision_knn, sensitivity_knn, specificity_knn = compute_metrics(y_test, y_test_pred_knn)

print("\nKNN Model Metrics:")
print(f"Test Accuracy: {accuracy_knn:.3f}, Precision: {precision_knn:.3f}, Sensitivity: {sensitivity_knn:.3f}, Specificity: {specificity_knn:.3f}")

#Gradient Boosting

xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_selected, y_train)

probs_xgb = xgb_model.predict_proba(X_test_selected)[:, 1]
y_test_pred_xgb = (probs_xgb > 0.5).astype(int)

accuracy_xgb, precision_xgb, sensitivity_xgb, specificity_xgb = compute_metrics(y_test, y_test_pred_xgb)

print("\nXGBoost Model Metrics:")
print(f"Test Accuracy: {accuracy_xgb:.3f}, Precision: {precision_xgb:.3f}, Sensitivity: {sensitivity_xgb:.3f}, Specificity: {specificity_xgb:.3f}")


# Compute ROC curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, probs)
fpr_svm, tpr_svm, _ = roc_curve(y_test, probs_svm)
fpr_nn, tpr_nn, _ = roc_curve(y_test, probs_nn)

# Compute AUCs
auc_rf = roc_auc_score(y_test, probs)
auc_svm = roc_auc_score(y_test, probs_svm)
auc_nn = roc_auc_score(y_test, probs_nn)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc_svm:.3f})")
plt.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC = {auc_nn:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: RF vs SVM vs NN')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
from sklearn.utils import shuffle

def y_randomization_test(model, X_train, y_train, X_test, y_test, n_iter=10, model_name="Model"):
    auc_scores = []

    for i in range(n_iter):
        # Shuffle the labels
        y_train_shuffled = shuffle(y_train, random_state=i)
        
        # Re-train model on shuffled labels
        model.fit(X_train, y_train_shuffled)
        probs = model.predict_proba(X_test)[:, 1]

        # Evaluate on the actual test set
        auc = roc_auc_score(y_test, probs)
        auc_scores.append(auc)
        print(f"{model_name} - Iteration {i+1}: AUC = {auc:.3f}")
    
    print(f"\nAverage AUC after Y-randomization ({model_name}): {np.mean(auc_scores):.3f}")
    return auc_scores

# Run Y-randomization for Random Forest
print("\n--- Y-Randomization: Random Forest ---")
_ = y_randomization_test(RandomForestClassifier(n_estimators=100, random_state=42),
                         X_train_selected, y_train, X_test_selected, y_test,
                         n_iter=10, model_name="Random Forest")

# Run Y-randomization for SVM
print("\n--- Y-Randomization: SVM ---")
_ = y_randomization_test(SVC(probability=True, random_state=42),
                         X_train_selected, y_train, X_test_selected, y_test,
                         n_iter=10, model_name="SVM")

def y_randomization_test_ann(X_train, y_train, X_test, y_test, n_iter=10):
    auc_scores = []

    for i in range(n_iter):
        # Shuffle the labels
        y_train_shuffled = shuffle(y_train, random_state=i)

        # Build a new ANN model for each iteration
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train_shuffled, epochs=30, batch_size=16, verbose=0)

        # Get probability predictions
        probs = model.predict(X_test).flatten()
        auc = roc_auc_score(y_test, probs)
        auc_scores.append(auc)

        print(f"ANN - Iteration {i+1}: AUC = {auc:.3f}")

    print(f"\nAverage AUC after Y-randomization (ANN): {np.mean(auc_scores):.3f}")
    return auc_scores
print("\n--- Y-Randomization: Random Forest ---")
auc_rf_rand = y_randomization_test(RandomForestClassifier(n_estimators=100, random_state=42),
                                   X_train_selected, y_train, X_test_selected, y_test,
                                   n_iter=10, model_name="Random Forest")

print("\n--- Y-Randomization: SVM ---")
auc_svm_rand = y_randomization_test(SVC(probability=True, random_state=42),
                                    X_train_selected, y_train, X_test_selected, y_test,
                                    n_iter=10, model_name="SVM")

print("\n--- Y-Randomization: Neural Network (ANN) ---")
auc_ann_rand = y_randomization_test_ann(X_train_selected, y_train, X_test_selected, y_test, n_iter=10)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), auc_rf_rand, marker='o', label='Random Forest')
plt.plot(range(1, 11), auc_svm_rand, marker='s', label='SVM')
plt.plot(range(1, 11), auc_ann_rand, marker='^', label='ANN')
plt.axhline(y=0.5, linestyle='--', color='gray', label='Chance')
plt.title("Y-Randomization AUC Scores (10 Iterations)")
plt.xlabel("Iteration")
plt.ylabel("AUC Score")
plt.ylim(0.4, 0.65)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.utils import resample
from scipy.stats import ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt

n_bootstrap = 1000
auc_rf_list = []
auc_svm_list = []
auc_ann_list = []

# Resample test set and compute AUCs
for _ in range(n_bootstrap):
    X_test_resampled, y_test_resampled = resample(X_test_selected, y_test, replace=True)

    # Predict probabilities for each model
    probs_rf = rf_model.predict_proba(X_test_resampled)[:, 1]
    probs_svm = svm_model.predict_proba(X_test_resampled)[:, 1]
    probs_ann = model.predict(X_test_resampled).flatten()

    # Compute AUCs
    auc_rf_list.append(roc_auc_score(y_test_resampled, probs_rf))
    auc_svm_list.append(roc_auc_score(y_test_resampled, probs_svm))
    auc_ann_list.append(roc_auc_score(y_test_resampled, probs_ann))

# Paired T-tests
print("\n--- Bootstrap and Student's T-test ---")
# RF vs SVM
t_rf_svm, p_rf_svm = ttest_rel(auc_rf_list, auc_svm_list)
print(f"RF vs SVM: t = {t_rf_svm:.4f}, p = {p_rf_svm:.4f}")

# RF vs ANN
t_rf_ann, p_rf_ann = ttest_rel(auc_rf_list, auc_ann_list)
print(f"RF vs ANN: t = {t_rf_ann:.4f}, p = {p_rf_ann:.4f}")

# ANN vs SVM
t_ann_svm, p_ann_svm = ttest_rel(auc_ann_list, auc_svm_list)
print(f"ANN vs SVM: t = {t_ann_svm:.4f}, p = {p_ann_svm:.4f}")

# Interpretation
if p_rf_ann < 0.05:
    print("→ Statistically significant difference between RF and ANN")
else:
    print("→ No significant difference between RF and ANN")

# KDE Plot
plt.figure(figsize=(10,6))
sns.kdeplot(auc_rf_list, label="Random Forest", fill=True)
sns.kdeplot(auc_svm_list, label="SVM", fill=True)
sns.kdeplot(auc_ann_list, label="ANN", fill=True)
plt.xlabel('AUC')
plt.ylabel('Density')
plt.title('Bootstrap Distribution of AUCs (RF vs SVM vs ANN)')
plt.legend()
plt.grid(True)
plt.show()


# Plot feature importance for Random Forest
importances = rf_model.feature_importances_
features = X_train_selected.columns

# Sort
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.grid(True)
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_test_pred)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Random Forest')
plt.show()

# SVM Confusion Matrix
cm_svm = confusion_matrix(y_test, y_test_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp_svm.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - SVM')
plt.show()

# ANN
cm_ann = confusion_matrix(y_test, y_test_pred_nn)
disp_ann = ConfusionMatrixDisplay(confusion_matrix=cm_ann)
disp_ann.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - ANN')
plt.show()

from sklearn.model_selection import learning_curve

# Random Forest Learning Curve
train_sizes, train_scores_rf, test_scores_rf = learning_curve(
    rf_model, X_train_selected, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores_rf, axis=1), 'o-', label='Training Accuracy')
plt.plot(train_sizes, np.mean(test_scores_rf, axis=1), 'o-', label='Validation Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve - Random Forest')
plt.legend()
plt.grid(True)
plt.show()

# SVM Learning Curve
train_sizes, train_scores_svm, test_scores_svm = learning_curve(
    svm_model, X_train_selected, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores_svm, axis=1), 'o-', label='Training Accuracy')
plt.plot(train_sizes, np.mean(test_scores_svm, axis=1), 'o-', label='Validation Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve - SVM')
plt.legend()
plt.grid(True)
plt.show()
from tensorflow.keras.optimizers import Adam

# === ANN Training and Learning Curve ===
# Split the same X_selected used above]
X_selected = X_resampled[important_features]
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)


model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=0
)

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Learning Curve - ANN')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
