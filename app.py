import uvicorn
from fastapi import FastAPI
from userinfo import UserInfo
import numpy as np
import pickle
import pandas as pd

app = FastAPI()

# Load model and preprocessors
with open("xgboost_classifier_model.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

# Helper function for preprocessing
def preprocess_input(data: dict):
    weight = float(data['weight'])
    height = float(data['height'])
    bmi = float(data['bmi'])
    body_fat_percentage = float(data['body_fat_percentage'])
    gender = data['gender']
    age = float(data['age'])

    # Encode gender
    gender_map = {'male': 1, 'female': 0}
    gender_encoded = gender_map.get(gender.lower(), 0)  # Make sure gender is lowercase

    # Scale features
    features = [[weight, height, bmi, body_fat_percentage, age]]
    features_scaled = scaler.transform(features)[0]

    # Create final input list
    input_data = list(features_scaled)[:4] + [gender_encoded, features_scaled[4]]

    return input_data

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/predict')
def predict_plan(data: UserInfo):
    input_data = preprocess_input(data.dict())
    print("Preprocessed input:", input_data)

    prediction = classifier.predict([input_data])
    return {'prediction': int(prediction[0])}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
