import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "disease_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "Training.csv")
TEST_CSV_PATH = os.path.join(BASE_DIR, "Testing.csv")

# 2. Load model and scaler
clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Model and scaler loaded successfully!")

# 3. Load training data to get features
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
feature_columns = train_df.drop(columns=["prognosis"]).columns

# 4. Load and preprocess test data
test_df = pd.read_csv(TEST_CSV_PATH)
test_df = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]

# Check missing columns 
missing_cols = set(feature_columns) - set(test_df.columns)
if missing_cols:
    raise ValueError(f"Test CSV is missing these columns: {missing_cols}")

X_test = test_df[feature_columns]
X_test_scaled = scaler.transform(X_test)
y_test = test_df["prognosis"]

# 5. Predict and evaluate
y_pred = clf.predict(X_test_scaled)

print("\n--- Test Performance ---")
test_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", test_acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Confusion matrix visualization [Added]
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Test Data")
plt.show()
