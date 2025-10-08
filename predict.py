import os
import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(_file_))
MODEL_PATH = os.path.join(BASE_DIR, "disease_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "Training.csv")

print("Looking for model at:", MODEL_PATH)
print("Looking for training CSV at:", TRAIN_CSV_PATH)

clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Model and scaler loaded successfully!")

         