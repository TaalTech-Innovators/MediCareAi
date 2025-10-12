import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "Training.csv")
MODEL_PATH = os.path.join(BASE_DIR, "disease_model.pkl")

print("Loading training data from:", TRAIN_CSV_PATH)
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]

# 2. Exploratory Data Analysis 
print("\n--- Dataset Summary ---")
print(train_df.describe())

print("\n--- Class Distribution ---")
print(train_df['prognosis'].value_counts())

# 3. Preprocessing
# Handle missing values
if train_df.isnull().sum().sum() > 0:
    train_df.fillna(0, inplace=True) 

X = train_df.drop(columns=["prognosis"])
y = train_df["prognosis"]

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split training and testing for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y  
)

# 5. Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2'], 
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

# 6. Model evaluation
y_train_pred = best_clf.predict(X_train)
y_test_pred = best_clf.predict(X_test)

# Training metrics
train_acc = accuracy_score(y_train, y_train_pred)
print("\n--- Training Performance ---")
print("Accuracy:", train_acc)
print("Classification Report:\n", classification_report(y_train, y_train_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

# Testing metrics
test_acc = accuracy_score(y_test, y_test_pred)
print("\n--- Testing Performance ---")
print("Accuracy:", test_acc)
print("Classification Report:\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix (Test Data):")
print(confusion_matrix(y_test, y_test_pred))

# Overfitting check 
print(f"\nOverfitting (Train - Test Accuracy): {train_acc - test_acc:.4f}")

# 7. Cross-validation
cv_scores = cross_val_score(best_clf, X_scaled, y, cv=5)
print(f"\n5-Fold Cross-validation Accuracy: {cv_scores.mean():.4f}")

# 8. Save model and scaler
joblib.dump(best_clf, MODEL_PATH)
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
print(f"\n✅ Model saved at: {MODEL_PATH}")
print(f"✅ Scaler saved at: {os.path.join(BASE_DIR, 'scaler.pkl')}")
