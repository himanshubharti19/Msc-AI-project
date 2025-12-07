import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_SIZE, SMOTE_RANDOM_STATE


def load_raw_dataset():
    """Fetch the UCI Myocardial Infarction Complications dataset."""
    dataset = fetch_ucirepo(id=579)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # If y is a DataFrame, take first column
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    return X, y


def preprocess_features(X):
    """Handle missing values, encode categoricals, scale features."""
    # Handle missing values
    X = X.fillna(X.median())

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled


def encode_target(y):
    """Label-encode target if needed."""
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)
    return y


def apply_smote(X, y):
    smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def train_test_split_stratified(X, y):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
