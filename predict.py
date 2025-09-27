import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt
from tabulate import tabulate

# -----------------------------
# 1. Load Model & Symptoms
# -----------------------------
clf = joblib.load("disease_model.pkl")

train_df = pd.read_csv("Training.csv")
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
symptoms = list(train_df.drop(columns=["prognosis"]).columns)

# -----------------------------
# 2. Random Symptom Selection
# -----------------------------
# Randomly pick 5 to 10 symptoms
selected_symptoms = random.sample(symptoms, k=random.randint(5, 10))

print(" Randomly Selected Symptoms for Prediction:")
print(", ".join(selected_symptoms))

# -----------------------------
# 3. Build Input Vector
# -----------------------------
input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

# -----------------------------
# 4. Prediction
# -----------------------------
probs = clf.predict_proba([input_data])[0]
disease_classes = clf.classes_

# Top 5 predictions
sorted_indices = probs.argsort()[::-1]
top5 = [(disease_classes[i], probs[i]) for i in sorted_indices[:5]]

# Display top 3
print("\n Predicted Disease (Most Likely):", top5[0][0])
print("\n Top 5 Predictions (with probabilities):\n")
print(tabulate(
    [(disease, f"{prob*100:.2f}%") for disease, prob in top5],
    headers=["Disease", "Probability"],
    tablefmt="fancy_grid"
))

# -----------------------------
# 5. Visualization
# -----------------------------
# Bar chart of top 5 predictions
labels = [d for d, _ in top5]
values = [p for _, p in top5]

plt.figure(figsize=(8, 5))
plt.barh(labels[::-1], [v*100 for v in values[::-1]], color='skyblue')
plt.xlabel("Probability (%)")
plt.title("Top 5 Disease Predictions")
plt.tight_layout()
plt.savefig("disease_prediction.png")
print("\n Prediction bar chart saved as 'disease_prediction.png'")
