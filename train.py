import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
from data_loader import load_data # type: ignore

print("Training script started...") #by Shivambu17
train_df=pd.read_csv("Training.csv") #by Shivambu17
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")] #by Shivambu17

X=train_df.drop(columns=["prognosis"]) #by Shivambu17
y=train_df["prognosis"] #by Shivambu17

print(" Training Random Forest model...")
