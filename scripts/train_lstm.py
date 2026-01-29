import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import json
from tqdm import tqdm
from trimandala.baselines.lstm import LSTMBaseline

try:
    from codecarbon import EmissionsTracker

    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False
    print("Warning: codecarbon not installed. Energy tracking disabled.")

import sys

if sys.platform == "darwin":
    HAS_CODECARBON = False
    print("CodeCarbon disabled on macOS (Requires sudo for powermetrics)")


class SequenceDataset(Dataset):
    def __init__(self, h5_file, seq_len=10):
        self.seq_len = seq_len
        self.file = h5_file
        with h5py.File(h5_file, "r") as f:
            self.pos = f["positions"][:]
            self.vel = f["velocities"][:]
            self.n_bodies = f.attrs["n_bodies"]

        self.pos_flat = self.pos.reshape(self.pos.shape[0], -1)
        self.vel_flat = self.vel.reshape(self.vel.shape[0], -1)
        self.data = np.concatenate([self.pos_flat, self.vel_flat], axis=1)  # (T, D)

        # Valid start indices
        self.valid_indices = range(len(self.data) - seq_len - 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Input: Sequence [t, t+1, ..., t+seq_len-1]
        # Target: Delta for LAST step [t+seq_len] - [t+seq_len-1]
        # Or should we predict delta at every step?
        # Better to predict delta at every step for dense supervision.

        # Segment: seq_len + 1
        segment = self.data[idx : idx + self.seq_len + 1]

        # X: [t ... t+seq_len-1]
        X = segment[:-1]

        # Y: Delta [t+1 ... t+seq_len] - [t ... t+seq_len-1]
        Y = segment[1:] - segment[:-1]

        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            Y, dtype=torch.float32
        )


def train():
    SEQ_LEN = 20
    BATCH_SIZE = 256
    EPOCHS = 20
    LR = 1e-3

    print("Loading Data (Sequences)...")
    train_dataset = SequenceDataset("data/train.h5", seq_len=SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Initializing LSTM...")
    model = LSTMBaseline(n_bodies=train_dataset.n_bodies, hidden_size=256, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5
    )
    criterion = nn.MSELoss()

    print(f"Training on {len(train_dataset)} sequences for {EPOCHS} epochs...")

    best_loss = float("inf")

    # Start energy tracking
    emissions = 0.0
    if HAS_CODECARBON:
        os.makedirs("metrics", exist_ok=True)
        tracker = EmissionsTracker(
            output_dir="metrics",
            output_file="lstm_training_emissions.csv",
            log_level="error",
        )
        tracker.start()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for x, y in pbar:
            optimizer.zero_grad()

            y_last = y[:, -1, :]

            pred, _ = model(x)
            loss = criterion(pred, y_last)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.2e}"})

        avg_loss = total_loss / len(train_loader)

        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}: Loss {avg_loss:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/lstm_baseline.pt")

    # Stop energy tracking
    if HAS_CODECARBON:
        emissions = tracker.stop()
        print(f"Training Complete. Carbon Emissions: {emissions:.4f} kg CO2eq")
    else:
        print("Training Complete.")

    print(f"Best Loss: {best_loss:.2e}")

    # Save training metadata with energy info
    metadata = {
        "model": "LSTMBaseline",
        "hidden_size": 256,
        "num_layers": 2,
        "seq_len": SEQ_LEN,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "best_loss": best_loss,
        "training_sequences": len(train_dataset),
        "emissions_kg_co2": emissions,
        "emissions_tracked": HAS_CODECARBON,
    }

    os.makedirs("models", exist_ok=True)
    with open("models/lstm_baseline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training metadata saved to models/lstm_baseline_metadata.json")


if __name__ == "__main__":
    train()
