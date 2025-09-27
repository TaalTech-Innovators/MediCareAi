import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# 1. Load Data
# -----------------------------
print("📥 Loading training data...")
train_df = pd.read_csv("Training.csv")
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]

X = train_df.drop(columns=["prognosis"])
y = train_df["prognosis"]

# -----------------------------
# 2. Train Model
# -----------------------------
print("🚀 Training Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# -----------------------------
# 3. Cross-validation
# -----------------------------
print("📊 Performing cross-validation...")
scores = cross_val_score(clf, X, y, cv=5)

print("\n✅ Cross-validation scores:", scores)
print(f"🎯 Average Accuracy: {scores.mean():.4f}")

# Plot accuracy scores
plt.figure(figsize=(7, 4))
plt.plot(range(1, 6), scores, marker='o', linestyle='-', color='blue')
plt.title("Cross-Validation Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("cross_validation_accuracy.png")
print("📈 Cross-validation plot saved as 'cross_validation_accuracy.png'")

# -----------------------------
# 4. Save Model
# -----------------------------
joblib.dump(clf, "disease_model.pkl")
print("💾 Model saved as 'disease_model.pkl'")
