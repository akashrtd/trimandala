import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
from tqdm import tqdm
from trimandala.baselines.lstm import LSTMBaseline

class SequenceDataset(Dataset):
    def __init__(self, h5_file, seq_len=10):
        self.seq_len = seq_len
        self.file = h5_file
        with h5py.File(h5_file, 'r') as f:
            self.pos = f['positions'][:]
            self.vel = f['velocities'][:]
            self.n_bodies = f.attrs['n_bodies']
            
        self.pos_flat = self.pos.reshape(self.pos.shape[0], -1)
        self.vel_flat = self.vel.reshape(self.vel.shape[0], -1)
        self.data = np.concatenate([self.pos_flat, self.vel_flat], axis=1) # (T, D)
        
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
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def train():
    SEQ_LEN = 20 # Longer history
    BATCH_SIZE = 256
    EPOCHS = 20
    LR = 1e-3
    
    print("Loading Data (Sequences)...")
    train_dataset = SequenceDataset("data/train.h5", seq_len=SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Initializing LSTM...")
    model = LSTMBaseline(n_bodies=train_dataset.n_bodies, hidden_size=256, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    
    print(f"Training on {len(train_dataset)} sequences for {EPOCHS} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for x, y in pbar:
            optimizer.zero_grad()
            # x: (Batch, Seq, Feat), y: (Batch, Seq, Feat)
            
            # Forward pass returns (Delta, Hidden)
            # Delta shape: (Batch, Seq, Feat)? No, my model currently outputs only last step.
            # I need to modify model or loop?
            # Actually, `LSTMBaseline.forward` in previous step takes x.
            # If x is 3D, it returns (Batch, Hidden) - only last step!
            # Wait, PyTorch LSTM returns output sequence.
            # I constructed `LSTMBaseline` to take `out[:, -1, :]` (Last Step).
            # This means I can only compute loss on the FINAL prediction.
            
            # Let's stick to "Last Step Prediction" for simplicity.
            # X: (Batch, Seq, Feat)
            # Y: (Batch, Seq, Feat) -> We only need Y[:, -1, :]
            
            y_last = y[:, -1, :]
            
            pred, _ = model(x) # Returns (Batch, Feat)
            loss = criterion(pred, y_last)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.2e}"})
            
        avg_loss = total_loss / len(train_loader)
        
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}: Loss {avg_loss:.2e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/lstm_baseline.pt")

    print(f"Training Complete. Best Loss: {best_loss:.2e}")

if __name__ == "__main__":
    train()
