import torch
import numpy as np

def predict_day8(model, last_seq):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = []

    current = last_seq.copy()

    for i in range(24):

        inp = torch.tensor(current[np.newaxis,:,:], dtype=torch.float32).to(device)

        pred = model(inp).detach().cpu().numpy()[0]

        predictions.append(pred)

        orbit = current[-1][4]

        hour = i % 24

        sin_time = np.sin(2*np.pi*hour/24)
        cos_time = np.cos(2*np.pi*hour/24)

        sin_time_12 = np.sin(2*np.pi*hour/12)
        cos_time_12 = np.cos(2*np.pi*hour/12)

        new_row = np.array([
            pred[0],
            pred[1],
            pred[2],
            pred[3],
            orbit,
            sin_time,
            cos_time,
            sin_time_12,
            cos_time_12
        ])

        current = np.vstack([current[1:], new_row])

    return np.array(predictions)