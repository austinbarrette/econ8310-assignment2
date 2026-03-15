import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# Helper functions
# -----------------------------
def add_features(df):
    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["hour"] = df["DateTime"].dt.hour
    df["dayofweek"] = df["DateTime"].dt.dayofweek
    df["month"] = df["DateTime"].dt.month

    base_exclude = {"id", "DateTime", "Total", "Discounts", "meal", "hour", "dayofweek", "month"}
    item_cols = [c for c in df.columns if c not in base_exclude]
    df["item_count"] = df[item_cols].sum(axis=1)

    return df

def tjurr(truth, pred):
    truth = list(truth)
    pred = list(pred)
    y1 = np.mean([y for i, y in enumerate(pred) if truth[i] == 1])
    y0 = np.mean([y for i, y in enumerate(pred) if truth[i] == 0])
    return y1 - y0

# -----------------------------
# Load and prepare training data
# -----------------------------
train_url = "https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3.csv"
train = pd.read_csv(train_url)
train = add_features(train)

y = train["meal"]
X = train.drop(columns=["meal", "DateTime", "id"])

# -----------------------------
# Validation split for tuning
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# Tune Random Forest + threshold
# -----------------------------
best_score = -1
best_params = None
best_threshold = 0.5

for n_estimators in [300, 500]:
    for max_depth in [None, 10, 20]:
        for min_samples_leaf in [1, 2, 5]:
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            )
            clf.fit(X_train, y_train)

            val_prob = clf.predict_proba(X_val)[:, 1]

            for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
                val_pred = [int(p >= threshold) for p in val_prob]
                score = tjurr(y_val, val_pred)

                if score > best_score:
                    best_score = score
                    best_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf
                    }
                    best_threshold = threshold

# -----------------------------
# Final model objects required by tests
# -----------------------------
model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_leaf=best_params["min_samples_leaf"],
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

modelFit = model.fit(X, y)

# -----------------------------
# Load and prepare test data
# -----------------------------
test_url = "https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3test.csv"
test = pd.read_csv(test_url)
test = add_features(test)

drop_cols = ["DateTime", "id"]
if "meal" in test.columns:
    drop_cols.append("meal")

X_test = test.drop(columns=drop_cols)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# -----------------------------
# Predictions required by tests
# -----------------------------
test_prob = modelFit.predict_proba(X_test)[:, 1]
pred = [int(p >= best_threshold) for p in test_prob]