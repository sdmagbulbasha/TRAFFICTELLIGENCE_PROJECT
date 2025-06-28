import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/sample_traffic_data.csv')

df['temp'] = df['temp'].fillna(df['temp'].mean())
df['rain'] = df['rain'].fillna(0)
df['snow'] = df['snow'].fillna(0)

df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

df['holiday'] = df['holiday'].fillna('None')
df['weather'] = df['weather'].fillna('Clear')

le_holiday = LabelEncoder()
le_weather = LabelEncoder()

df['holiday'] = le_holiday.fit_transform(df['holiday'])
df['weather'] = le_weather.fit_transform(df['weather'])

X = df[['holiday', 'temp', 'rain', 'snow', 'weather', 'hour', 'day_of_week', 'month']]
y = df['traffic_volume']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model trained successfully. RMSE: {rmse:.2f}")

os.makedirs('model', exist_ok=True)
with open('model/traffic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

os.makedirs('utils', exist_ok=True)
with open('utils/label_encoders.pkl', 'wb') as f:
    pickle.dump({'holiday': le_holiday, 'weather': le_weather}, f)
