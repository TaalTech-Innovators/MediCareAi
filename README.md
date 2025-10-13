# 🧠 MediCareAI – Disease Prediction Using Machine Learning

## 📘 About the Project
MediCareAI is an AI-based system that predicts possible diseases based on symptoms.  
It uses a **machine learning model (Random Forest Classifier)** trained on medical data containing 132 symptoms and 41 diseases.  
The goal is to help with **early detection of illnesses** and support doctors or users in identifying possible conditions quickly.

_________________________________________________________________________________________

## ⚙️ How It Works
There are **three main Python scripts** in this project:

1. **`train_model.py`**  
   - Trains the machine learning model using the dataset (`Training.csv`).  
   - Tunes the model for better accuracy using GridSearchCV.  
   - Saves the trained model (`disease_model.pkl`) and the scaler (`scaler.pkl`) for later use.  

2. **`test_model.py`**  
   - Loads the trained model and tests it with `Testing.csv`.  
   - Checks how accurate the model is.  
   - Shows a confusion matrix   

3. **`predict.py`**  
   - Loads the model and randomly selects symptoms.  
   - Predicts the **top 3 most likely diseases** based on those symptoms.  
   - Shows the prediction results and probabilities in a small bar chart.  

___________________________________________________________________________________________

## 🪜 Steps to Use the Project
1. **Make sure you have Python installed (version 3.x).**
2. **Install the required Python libraries** by running:
pip install pandas numpy scikit-learn matplotlib seaborn joblib

3. Train the model using:
python train_model.py

4. Test the model using:
python test_model.py

5. Predict diseases using:
python predict.py