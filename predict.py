import argparse
import pandas as pd
import joblib
import config

# -----------------------------
# 1. Load model and symptoms
# -----------------------------
clf = joblib.load("disease_model.pkl")

train_df = pd.read_csv("Training.csv")
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
symptoms = list(train_df.drop(columns=["prognosis"]).columns)
# -----------------------------
# 2. User input
# -----------------------------
print("\nEnter symptoms separated by commas (example: itching, skin_rash, nausea)")
user_input = input("Symptoms: ")

user_symptoms = [s.strip().lower().replace(" ", "_") for s in user_input.split(",")]
# -----------------------------

