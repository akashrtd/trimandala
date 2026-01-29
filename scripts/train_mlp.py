import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
from tqdm import tqdm
from trimandala.baselines.mlp import SimpleMLP

class NBodyDataset(Dataset):
    def __init__(self, h5_file, split='train'):
        self.file = h5_file
        with h5py.File(h5_file, 'r') as f:
            # Load all into memory (1M steps is small enough)
            self.pos = f['positions'][:]
            self.vel = f['velocities'][:]
            self.n_bodies = f.attrs['n_bodies']
            
        # Flatten state: (Steps, N, 3) -> (Steps, N*3)
        self.pos_flat = self.pos.reshape(self.pos.shape[0], -1)
        self.vel_flat = self.vel.reshape(self.vel.shape[0], -1)
        
        # X: [pos_t, vel_t]
        # Y: [pos_t+1, vel_t+1] - [pos_t, vel_t] (Delta)
        
        # Prepare Tensors
        self.X = np.concatenate([self.pos_flat[:-1], self.vel_flat[:-1]], axis=1)
        
        # Delta Pos
        d_pos = self.pos_flat[1:] - self.pos_flat[:-1]
        d_vel = self.vel_flat[1:] - self.vel_flat[:-1]
        self.Y = np.concatenate([d_pos, d_vel], axis=1)
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train():
    BATCH_SIZE = 1024
    EPOCHS = 50
    LR = 1e-3
    
    print("Loading Data...")
    train_dataset = NBodyDataset("data/train.h5")
    val_dataset = NBodyDataset("data/val.h5") # Use Val for scheduler
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Initializing Model (512 hidden units)...")
    model = SimpleMLP(n_bodies=train_dataset.n_bodies, hidden_size=512)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    print(f"Training on {len(train_dataset)} samples for {EPOCHS} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.2e}"})
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.2e} | Val Loss {avg_val_loss:.2e} | LR {optimizer.param_groups[0]['lr']:.2e}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/mlp_baseline.pt")

    print(f"Training Complete. Best Val Loss: {best_loss:.2e}")
    print("Model saved to models/mlp_baseline.pt")

if __name__ == "__main__":
    train()
