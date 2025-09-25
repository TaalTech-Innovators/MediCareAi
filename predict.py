
import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt
from tabulate import tabulate


clf = joblib.load("disease_model.pkl")

train_df = pd.read_csv("Training.csv")
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
symptoms = list(train_df.drop(columns=["prognosis"]).columns)


num_symptoms = 5  
user_symptoms = random.sample(symptoms, num_symptoms)

print("\nRandomly selected symptoms:", ", ".join(user_symptoms))

input_data = [1 if symptom in user_symptoms else 0 for symptom in symptoms]

probs = clf.predict_proba([input_data])[0]
disease_classes = clf.classes_

sorted_indices = probs.argsort()[::-1]
top3 = [(disease_classes[i], probs[i]) for i in sorted_indices[:3]]

print("\n Predicted Disease (Most Likely):", top3[0][0])
print("\n Top 3 Predictions:")
for disease, prob in top3:
    print(f" - {disease}: {prob*100:.2f}%")