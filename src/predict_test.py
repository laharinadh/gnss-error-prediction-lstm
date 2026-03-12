import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from sequence import create_sequences
from model import GNSSModel
from predict import predict_day8


def prepare_data(file):

    df = pd.read_csv(file)

    df = df.rename(columns={
        "utc_time":"time",
        "x_error (m)":"x",
        "y_error (m)":"y",
        "z_error (m)":"z",
        "satclockerror (m)":"clock"
    })

    df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values("time")

    df.set_index("time", inplace=True)
     # ADD ORBIT FEATURE
    if "GEO" in file:
        df["orbit"] = 0
    else:
        df["orbit"] = 1
    # periodic features
    hours = df.index.hour

    df["sin_time"] = np.sin(2*np.pi*hours/24)
    df["cos_time"] = np.cos(2*np.pi*hours/24)
    df["sin_time_12"] = np.sin(2*np.pi*hours/12)
    df["cos_time_12"] = np.cos(2*np.pi*hours/12)

    return df


def predict_file(model, file, output):

    df = prepare_data(file)

    data = df[[
    "x","y","z","clock",
    "orbit",
    "sin_time","cos_time",
    "sin_time_12","cos_time_12"
    ]].values

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    last_seq = data[-24:]

    pred = predict_day8(model, last_seq)

    pred_df = pd.DataFrame(pred, columns=["x","y","z","clock"])

    pred_df.to_csv(output, index=False)

    print("Saved:", output)


def run_all():

    model = GNSSModel()

    model.load_state_dict(torch.load("models/best_model.pth"))
    model.eval()

    predict_file(
        model,
        "data/DATA_GEO_Test.csv",
        "results/geo_prediction.csv"
    )

    predict_file(
        model,
        "data/DATA_MEO_Test.csv",
        "results/meo_prediction.csv"
    )

    predict_file(
        model,
        "data/DATA_MEO_Test2.csv",
        "results/meo_prediction2.csv"
    )


if __name__ == "__main__":
    run_all()