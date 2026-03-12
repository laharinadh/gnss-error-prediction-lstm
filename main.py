import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocess import load_data
from src.sequence import create_sequences
from src.model import GNSSModel
from src.train import train_model
from src.predict import predict_day8

os.makedirs("results", exist_ok=True)

df = load_data()

data = df[[
"x","y","z","clock",
"orbit",
"sin_time","cos_time",
"sin_time_12","cos_time_12"
]].values

scaler = StandardScaler()

data = scaler.fit_transform(data)

window = 48

X, y = create_sequences(data, window)

model = GNSSModel()

model = train_model(model, X, y)

last_seq = data[-window:]

pred = predict_day8(model, last_seq)

pred_df = pd.DataFrame(pred, columns=["x","y","z","clock"])

pred_df.to_csv("results/day8_prediction.csv", index=False)

print("Day 8 prediction saved")