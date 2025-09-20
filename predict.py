import pandas as pd
import joblib

# -----------------------------
# 1. Load model and symptoms
# -----------------------------
clf = joblib.load("disease_model.pkl")

train_df = pd.read_csv("Training.csv")

