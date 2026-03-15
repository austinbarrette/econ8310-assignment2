import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load training data
# -----------------------------
train_url = "https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3.csv"
train = pd.read_csv(train_url)

# Convert DateTime and create time features
train["DateTime"] = pd.to_datetime(train["DateTime"])
train["hour"] = train["DateTime"].dt.hour
train["dayofweek"] = train["DateTime"].dt.dayofweek
train["month"] = train["DateTime"].dt.month

# Define target and features
y = train["meal"]
X = train.drop(columns=["meal", "DateTime", "id"])

# -----------------------------
# Create and fit model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

modelFit = model.fit(X, y)

# -----------------------------
# Load test data
# -----------------------------
test_url = "https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3test.csv"
test = pd.read_csv(test_url)

# Convert DateTime and create same time features
test["DateTime"] = pd.to_datetime(test["DateTime"])
test["hour"] = test["DateTime"].dt.hour
test["dayofweek"] = test["DateTime"].dt.dayofweek
test["month"] = test["DateTime"].dt.month

# Drop columns not used in training
drop_cols = ["DateTime", "id"]
if "meal" in test.columns:
    drop_cols.append("meal")

X_test = test.drop(columns=drop_cols)

# Align test columns to training columns just in case
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# -----------------------------
# Generate predictions
# -----------------------------
pred = modelFit.predict(X_test).astype(int)