import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
print(" Loading training data...")
train_df = pd.read_csv("Training.csv")

# Drop any unnamed columns
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]

X = train_df.drop(columns=["prognosis"])
y = train_df["prognosis"]

print(" Training Random Forest model...")
