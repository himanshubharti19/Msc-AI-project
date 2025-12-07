import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from .data_preprocessing import (
    load_raw_dataset, preprocess_features,
    encode_target, apply_smote, train_test_split_stratified
)
from .config import (
    LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, RANDOM_STATE
)


def severity_score(prob):
    if prob < LOW_RISK_THRESHOLD:
        return "Low Risk"
    elif LOW_RISK_THRESHOLD <= prob < HIGH_RISK_THRESHOLD:
        return "Moderate Risk"
    else:
        return "High Risk"


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

    return accuracy, precision, sensitivity, specificity


def build_ann(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    # ---------- DATA ----------
    X, y = load_raw_dataset()
    X = preprocess_features(X)
    y = encode_target(y)
    X_resampled, y_resampled = apply_smote(X, y)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X_resampled, y_resampled)

    # ---------- FEATURE SELECTION ----------
    lasso = Lasso(alpha=0.01, random_state=RANDOM_STATE)
    lasso.fit(X_train, y_train)
    important_features = X.columns[lasso.coef_ != 0]
    print(f"Selected Features using LASSO: {important_features.tolist()}")

    X_train_sel = pd.DataFrame(X_train, columns=X.columns)[important_features]
    X_test_sel = pd.DataFrame(X_test, columns=X.columns)[important_features]

    # ---------- RANDOM FOREST ----------
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train_sel, y_train)
    probs_rf = rf_model.predict_proba(X_test_sel)[:, 1]
    y_pred_rf = (probs_rf > 0.5).astype(int)

    print("\nRandom Forest classification report:")
    print(classification_report(y_test, y_pred_rf))

    acc_tr_rf, prec_tr_rf, sen_tr_rf, spec_tr_rf = compute_metrics(
        y_train, rf_model.predict(X_train_sel)
    )
    acc_te_rf, prec_te_rf, sen_te_rf, spec_te_rf = compute_metrics(
        y_test, y_pred_rf
    )
    print(f"\nRandom Forest – Train: acc={acc_tr_rf:.3f}, prec={prec_tr_rf:.3f}, "
          f"recall={sen_tr_rf:.3f}, spec={spec_tr_rf:.3f}")
    print(f"Random Forest – Test:  acc={acc_te_rf:.3f}, prec={prec_te_rf:.3f}, "
          f"recall={sen_te_rf:.3f}, spec={spec_te_rf:.3f}")

    # Severity labels (RF)
    severity_labels = [severity_score(p) for p in probs_rf]
    print("\nSample RF severity scores:")
    for i in range(10):
        print(f"Sample {i+1}: prob={probs_rf[i]:.3f}, severity={severity_labels[i]}")

    # ---------- ANN ----------
    ann = build_ann(X_train_sel.shape[1])
    history = ann.fit(
        X_train_sel, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test_sel, y_test),
        verbose=0
    )
    probs_ann = ann.predict(X_test_sel).flatten()
    y_pred_ann = (probs_ann > 0.5).astype(int)

    acc_tr_ann, prec_tr_ann, sen_tr_ann, spec_tr_ann = compute_metrics(
        y_train, (ann.predict(X_train_sel).flatten() > 0.5).astype(int)
    )
    acc_te_ann, prec_te_ann, sen_te_ann, spec_te_ann = compute_metrics(
        y_test, y_pred_ann
    )
    print(f"\nANN – Train: acc={acc_tr_ann:.3f}, prec={prec_tr_ann:.3f}, "
          f"recall={sen_tr_ann:.3f}, spec={spec_tr_ann:.3f}")
    print(f"ANN – Test:  acc={acc_te_ann:.3f}, prec={prec_te_ann:.3f}, "
          f"recall={sen_te_ann:.3f}, spec={spec_te_ann:.3f}")

    # ---------- SVM (and optionally other models) ----------
    svm_model = SVC(probability=True, random_state=RANDOM_STATE)
    svm_model.fit(X_train_sel, y_train)
    probs_svm = svm_model.predict_proba(X_test_sel)[:, 1]
    y_pred_svm = (probs_svm > 0.5).astype(int)

    acc_te_svm, prec_te_svm, sen_te_svm, spec_te_svm = compute_metrics(y_test, y_pred_svm)
    print(f"\nSVM – Test: acc={acc_te_svm:.3f}, prec={prec_te_svm:.3f}, "
          f"recall={sen_te_svm:.3f}, spec={spec_te_svm:.3f}")

    # ---------- ROC CURVES ----------
    fpr_rf, tpr_rf, _ = roc_curve(y_test, probs_rf)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, probs_svm)
    fpr_ann, tpr_ann, _ = roc_curve(y_test, probs_ann)

    auc_rf = roc_auc_score(y_test, probs_rf)
    auc_svm = roc_auc_score(y_test, probs_svm)
    auc_ann = roc_auc_score(y_test, probs_ann)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})")
    plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={auc_svm:.3f})")
    plt.plot(fpr_ann, tpr_ann, label=f"ANN (AUC={auc_ann:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
