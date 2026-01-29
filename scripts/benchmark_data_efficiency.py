import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
from tqdm import tqdm
from trimandala.baselines.mlp import SimpleMLP
from trimandala.arena import Arena
import matplotlib.pyplot as plt


# Reuse Dataset but with limit
class NBodyDataset(Dataset):
    def __init__(self, h5_file, limit: Optional[int] = None):
        self.file = h5_file
        with h5py.File(h5_file, "r") as f:
            total = len(f["time"])
            if limit:
                # Assuming steps=1, data is interleaved?
                # Our previous loader loaded everything.
                # Let's load slice.
                self.pos = np.asarray(f["positions"][: limit + 1])
                self.vel = np.asarray(f["velocities"][: limit + 1])
            else:
                self.pos = np.asarray(f["positions"][:])
                self.vel = np.asarray(f["velocities"][:])
            self.n_bodies = int(f.attrs["n_bodies"])

        self.pos_flat = self.pos.reshape(self.pos.shape[0], -1)
        self.vel_flat = self.vel.reshape(self.vel.shape[0], -1)

        self.X = np.concatenate([self.pos_flat[:-1], self.vel_flat[:-1]], axis=1)
        d_pos = self.pos_flat[1:] - self.pos_flat[:-1]
        d_vel = self.vel_flat[1:] - self.vel_flat[:-1]
        self.Y = np.concatenate([d_pos, d_vel], axis=1)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_model(samples_count: int):
    print(f"\n--- Training on {samples_count} samples ---")
    BATCH_SIZE = 64
    EPOCHS = 10  # Fast training
    LR = 1e-3

    train_dataset = NBodyDataset("data/train.h5", limit=samples_count)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleMLP(
        n_bodies=train_dataset.n_bodies, hidden_size=256
    )  # Smaller model for speed
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    return model


def main():
    print("=== Trimandala Benchmark: Data Efficiency ===")

    subsets = [100, 1000, 10000]
    scores = []

    arena = Arena("data/val.h5")

    for n in subsets:
        model = train_model(n)
        results = arena.run_track_a(model.predict, t_pred=100)
        print(f"Samples: {n} | TES: {results['tes_score']:.2f}")
        scores.append(results["tes_score"])

    print("\n--- Efficiency Curve ---")
    for n, s in zip(subsets, scores):
        print(f"{n}: {s:.2f}")

    # Plot
    plt.figure()
    plt.semilogx(subsets, scores, marker="o")
    plt.title("Data Efficiency Curve (MLP)")
    plt.xlabel("Training Samples")
    plt.ylabel("TES Score")
    plt.grid(True)
    plt.savefig("data_efficiency.png")
    print("Saved plot to data_efficiency.png")


if __name__ == "__main__":
    main()
