import numpy as np

def create_sequences(data, window):

    X = []
    y = []

    for i in range(len(data) - window):

        X.append(data[i:i+window])
        y.append(data[i+window][:4])

    return np.array(X), np.array(y)