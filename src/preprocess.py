import pandas as pd
import numpy as np

def load_data():

    geo = pd.read_csv("data/DATA_GEO_Train.csv")
    meo1 = pd.read_csv("data/DATA_MEO_Train.csv")
    meo2 = pd.read_csv("data/DATA_MEO_Train2.csv")

    meo = pd.concat([meo1, meo2])

    geo["orbit"] = 0
    meo["orbit"] = 1

    df = pd.concat([geo, meo])

    df = df.rename(columns={
        "utc_time":"time",
        "x_error (m)":"x",
        "y_error (m)":"y",
        "z_error (m)":"z",
        "satclockerror (m)":"clock"
    })

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    df = df.drop_duplicates(subset="time")

    df.set_index("time", inplace=True)

    df = df.resample("1h").interpolate()

    hours = df.index.hour

    df["sin_time"] = np.sin(2*np.pi*hours/24)
    df["cos_time"] = np.cos(2*np.pi*hours/24)

    df["sin_time_12"] = np.sin(2*np.pi*hours/12)
    df["cos_time_12"] = np.cos(2*np.pi*hours/12)

    return df