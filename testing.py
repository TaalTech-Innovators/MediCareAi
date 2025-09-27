import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

# -----------------------------
# 1. Load Model & Data
# -----------------------------
clf = joblib.load("disease_model.pkl")

train_df = pd.read_csv("Training.csv")
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
feature_columns = train_df.drop(columns=["prognosis"]).columns

test_df = pd.read_csv("Testing.csv")
test_df = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]

X_test = test_df[feature_columns]
y_test = test_df["prognosis"]

# -----------------------------
# 2. Predict
# -----------------------------
y_pred = clf.predict(X_test)

# -----------------------------
# 3. Evaluate
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {accuracy:.4f}\n")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Save classification report as table
report_df.to_csv("classification_report.csv")
print("📄 Classification report saved to 'classification_report.csv'")

# Display bar chart of class-wise F1-score
f1_scores = report_df.iloc[:-3]["f1-score"]

plt.figure(figsize=(10, 5))
f1_scores.sort_values().plot(kind='barh', color='green')
plt.title("F1-Scores per Class")
plt.xlabel("F1-Score")
plt.tight_layout()
plt.savefig("f1_scores.png")
print("📈 F1-score bar chart saved as 'f1_scores.png'")

# Confusion Matrix (Optional)
# ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
# plt.savefig("confusion_matrix.png")
