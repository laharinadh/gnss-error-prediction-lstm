import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

def train_model(model, X, y):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    os.makedirs("models", exist_ok=True)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, y)

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    loss_fn = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(150):

        model.train()

        total_loss = 0

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            pred = model(batch_x)

            loss = loss_fn(pred, batch_y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print("Epoch:",epoch,"Loss:",avg_loss)

        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(model.state_dict(),"models/best_model.pth")

            print("✔ Best model saved")

    print("Best training loss:",best_loss)

    return model