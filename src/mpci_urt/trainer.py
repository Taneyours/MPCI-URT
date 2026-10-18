from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from .metrics import classification_accuracy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def resolve_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)

def train_model(
    model,
    train_loader,
    edge_index,
    device,
    epochs,
    learning_rate,
    weight_decay,
    alpha,
    checkpoint_path,
):
    model = model.to(device)
    edge_index = edge_index.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss().to(device)

    history = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data, edge_index)

            loss_pred = criterion(outputs, batch_data.y)
            loss_traj = criterion(batch_data.trajectory_scores, batch_data.y)
            loss = alpha * loss_pred + (1.0 - alpha) * loss_traj

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_losses))
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"Epoch {epoch + 1:03d} | loss={mean_loss:.6f}")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # The original script's "best_model" variable referenced the live model,
    # so the saved file effectively tracked the current/final model.
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    pd.DataFrame(history).to_csv(
        checkpoint_path.with_suffix(".history.csv"), index=False
    )
    return model

@torch.no_grad()
def evaluate_model(
    model, test_loader, edge_index, device, predictions_path=None
):
    model = model.to(device)
    edge_index = edge_index.to(device)
    model.eval()

    y_true_all = []
    y_pred_all = []

    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        logits = model(batch_data, edge_index)
        pred = logits.argmax(dim=-1)

        y_true_all.extend(batch_data.y.detach().cpu().numpy().tolist())
        y_pred_all.extend(pred.detach().cpu().numpy().tolist())

    acc = classification_accuracy(y_true_all, y_pred_all)
    print(f"Accuracy: {acc * 100:.2f}%")

    if predictions_path is not None:
        predictions_path = Path(predictions_path)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"pred": y_pred_all, "true": y_true_all}
        ).to_csv(predictions_path, index=False)

    return {"accuracy": acc}
