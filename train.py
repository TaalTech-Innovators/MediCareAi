import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

import config
from data_loader import load_data
from model import build_model, save_model

def main(data_file):
    df = load_data(data_file, config.TARGET_COLUMN)
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN].astype(str)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_encoded
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    save_model(model, config.MODELS_DIR / "model.joblib")
    joblib.dump(le, config.MODELS_DIR / "label_encoder.joblib")
    print("✅ Model and encoder saved in 'models/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=config.TRAIN_FILE, help="Path to training CSV")
    args = parser.parse_args()
    main(args.data)
