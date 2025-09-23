
import argparse #by kamo
import joblib #by kgotso (corrected)
from sklearn.metrics import classification_report, accuracy_score #by kgotso (corrected)
import config                   # by Kamogelo
from data_loader import load_data        # by Kamogelo
from model import load_model             # by Kamogelo

def main(model_path, test_file): #by Angela
model = load_model(model_path)  #by Angela
le = joblib.load(config.MODELS_DIR / "label_encoder.joblib") #by Angela
df = load_data(test_file, config.TARGET_COLUMN) #by kamo
X_test = df.drop(columns = [config.TARGET_COLUMN]) #by Angela
y_test = le.transform(df[config.TARGET_COLUMN].astype(str)) #by Kgotso