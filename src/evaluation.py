import pandas as pd
import numpy as np
from scipy.stats import shapiro, probplot
import matplotlib.pyplot as plt


# ---------------------------
# Load and prepare actual data
# ---------------------------
def load_actual(file):

    df = pd.read_csv(file)

    df = df.rename(columns={
        "x_error (m)": "x",
        "y_error (m)": "y",
        "z_error (m)": "z",
        "satclockerror (m)": "clock"
    })

    return df[["x","y","z","clock"]]


# ---------------------------
# Remove GNSS spikes
# ---------------------------
def clean_residuals(residual):

    residual = residual[np.isfinite(residual)]

    if len(residual) == 0:
        return residual

    q1 = np.percentile(residual, 25)
    q3 = np.percentile(residual, 75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    residual = residual[(residual >= lower) & (residual <= upper)]

    # remove mean bias
    residual = residual - np.mean(residual)

    return residual


# ---------------------------
# Evaluate dataset
# ---------------------------
def evaluate_dataset(actual_file, pred_file, name):

    actual = load_actual(actual_file)

    pred = pd.read_csv(pred_file)

    pred = pred[["x","y","z","clock"]]

    # match lengths
    min_len = min(len(actual), len(pred))

    actual = actual.iloc[:min_len]
    pred = pred.iloc[:min_len]

    # convert to numpy
    actual_vals = actual.values
    pred_vals = pred.values

    # ---------------------------
    # RMSE calculation
    # ---------------------------
    rmse = np.sqrt(np.mean((actual_vals - pred_vals) ** 2))

    residual = actual_vals - pred_vals
    residual = residual.flatten()

    residual = clean_residuals(residual)

    if len(residual) < 3:
        print("\nDataset:", name)
        print("Not enough samples for Shapiro test")
        return None

    stat, p = shapiro(residual)

    print("\n==============================")
    print("Dataset:", name)
    print("RMSE:", rmse)
    print("Samples:", len(residual))
    print("Shapiro p-value:", p)

    if p > 0.05:
        print("Residuals follow normal distribution ✔")
    else:
        print("Residuals NOT normal ❌")

    # ---------------------------
    # Histogram
    # ---------------------------
    plt.figure()
    plt.hist(residual, bins=30)
    plt.title(f"{name} Residual Histogram")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    # ---------------------------
    # Q-Q Plot
    # ---------------------------
    plt.figure()
    probplot(residual, dist="norm", plot=plt)
    plt.title(f"{name} Q-Q Plot")

    plt.show()

    return p


# ---------------------------
# Run evaluation
# ---------------------------
def run_evaluation():

    results = {}

    results["GEO"] = evaluate_dataset(
        "data/DATA_GEO_Test.csv",
        "results/geo_prediction.csv",
        "GEO"
    )

    results["MEO"] = evaluate_dataset(
        "data/DATA_MEO_Test.csv",
        "results/meo_prediction.csv",
        "MEO"
    )

    print("\n==============================")
    print("FINAL SUMMARY")
    print("==============================")

    for k,v in results.items():

        if v is None:
            print(f"{k} -> No result")
        else:
            status = "PASS ✔" if v > 0.05 else "FAIL ❌"
            print(f"{k}  p-value = {v:.5f}  -> {status}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    run_evaluation()