import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load and prepare dataset
df = pd.read_csv("parkinsons.csv")
df.drop(['name'], axis=1, inplace=True)

X = df.drop('status', axis=1)
y = df['status']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Define individual models
lr = LogisticRegression(max_iter=1000, random_state=42)
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.1,
    scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1],
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=0
)
lgb = LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

# Voting classifier
voting = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('lgb', lgb),
        ('lr', lr)
    ],
    voting='soft',
    weights=[1, 3, 2]
)

# Train and evaluate
voting.fit(X_train_res, y_train_res)
y_pred = voting.predict(X_test_scaled)

print("\n✅ Final Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and scaler
joblib.dump(scaler, "scaler.pkl")
joblib.dump(voting, "parkinson_model.pkl")
print("\nModel and scaler saved successfully.")



