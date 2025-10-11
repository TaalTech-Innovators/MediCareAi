import os
import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt

# 1. Set base folder and paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "disease_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "Training.csv")

print("Looking for model at:", MODEL_PATH)
print("Looking for training CSV at:", TRAIN_CSV_PATH)

# 2. Load model and scaler
clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Model and scaler loaded successfully!")

# 3. Load training data to get feature names
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
X = train_df.drop(columns=["prognosis"])
symptom_columns = X.columns  # ✅ This ensures names align with the model

# 4. Random symptom selection
num_symptoms = 5  # Change the number if needed
selected_symptoms = random.sample(list(symptom_columns), num_symptoms)
print("\nRandomly selected symptoms:", ", ".join(selected_symptoms))

# 5. Build input as a DataFrame instead of raw NumPy array
input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_columns]
input_df = pd.DataFrame([input_vector], columns=symptom_columns)

# ✅ Scale with DataFrame to avoid warnings
input_scaled = scaler.transform(input_df)

# 6. Make predictions
probs = clf.predict_proba(input_scaled)[0]
disease_classes = clf.classes_

# Sort by probability (highest first)
sorted_indices = probs.argsort()[::-1]
top3 = [(disease_classes[i], probs[i]) for i in sorted_indices[:3]]

print("\nPredicted Disease (Most Likely):", top3[0][0])
print("\nTop 3 Predictions:")
for disease, prob in top3:
    print(f" - {disease}: {prob*100:.2f}%")

# -----------------------------
# 7. Visualization (Top-3 bar chart)
# -----------------------------
diseases, probs_list = zip(*top3)
plt.figure(figsize=(10,5))
plt.bar(diseases, probs_list)

# ✅ Main title
plt.title("Top 3 Predicted Diseases", pad=20)

# ✅ Subtitle with symptoms
plt.suptitle("Symptoms: " + ", ".join(selected_symptoms), fontsize=10, y=0.97)

# ✅ Adjust layout
plt.subplots_adjust(top=0.85)
plt.ylabel("Probability")
plt.tight_layout()
plt.show()
