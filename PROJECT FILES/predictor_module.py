import pickle
import numpy as np
import os
from datetime import datetime

model_path = os.path.join("model", "traffic_model.pkl")
encoder_path = os.path.join("utils", "label_encoders.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(encoder_path):
    raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(encoder_path, "rb") as f:
    encoders = pickle.load(f)

def predict_traffic(holiday, temp, rain, snow, weather, date, time):
    try:
        holiday_encoded = encoders['holiday'].transform([holiday])[0]
        weather_encoded = encoders['weather'].transform([weather])[0]

        date_obj = datetime.strptime(date, "%Y-%m-%d")
        hour = int(time.split(":")[0])
        day_of_week = date_obj.weekday()
        month = date_obj.month

        features = np.array([[holiday_encoded, float(temp), float(rain), float(snow),
                              weather_encoded, hour, day_of_week, month]])

        prediction = model.predict(features)[0]
        return round(prediction)

    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")
