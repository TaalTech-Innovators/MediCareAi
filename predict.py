import pandas as pd
import joblib

# -----------------------------
# 1. Load model and symptoms
# -----------------------------
clf = joblib.load("disease_model.pkl")

train_df = pd.read_csv("Training.csv")
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
symptoms = list(train_df.drop(columns=["prognosis"]).columns)
# -----------------------------
# 2. User input

