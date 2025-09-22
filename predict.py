import argparse
import pandas as pd
import joblib
import config
from data_loader import load_data
from model import load_model

def main(model_path, input_file, output_file):
    model = load_model(model_path)
le = joblib.load(config.MODELS_DIR / "label_encoder.joblib")

df = load_data(input_file)
if config.TARGET_COLUMN in df.columns:
 df = df.drop(columns=[config.TARGET_COLUMN])





