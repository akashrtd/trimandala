import torch
import torch.nn as nn
import numpy as np

class LSTMBaseline(nn.Module):
    """
    Recurrent Neural Network for N-Body trajectory prediction.
    Architecture:
    Input (N*6) -> Project -> LSTM(hidden) -> Head -> Delta(N*6)
    """
    def __init__(self, n_bodies=3, hidden_size=256, num_layers=2):
        super().__init__()
        self.n = n_bodies
        input_dim = n_bodies * 6
        output_dim = input_dim
        
        # Projection layer
        self.encoder = nn.Linear(input_dim, hidden_size)
        
        # RNN Core
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
        
    def forward(self, x, h=None):
        # x shape: (Batch, Seq, Feat) or (Batch, Feat)
        # If input is (Batch, Feat), unsqueeze seq dim
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        emb = torch.relu(self.encoder(x))
        out, h_next = self.lstm(emb, h)
        
        # Take last step output
        # out shape: (Batch, Seq, Hidden)
        last_step = out[:, -1, :]
        
        delta = self.decoder(last_step)
        return delta, h_next

    def predict(self, pos, vel, dt, steps):
        """
        Rollout API.
        Since we trained one-step-ahead, we can feed back the prediction.
        Ideally we should train on sequence, but for baseline we stick to iterative.
        """
        # Prepare initial state
        pos_t = torch.tensor(pos, dtype=torch.float32).flatten().unsqueeze(0)
        vel_t = torch.tensor(vel, dtype=torch.float32).flatten().unsqueeze(0)
        state = torch.cat([pos_t, vel_t], dim=1) # (1, 18)
        
        self.eval()
        hidden = None
        
        with torch.no_grad():
            for _ in range(steps):
                # Predict delta
                delta, hidden = self.forward(state, hidden)
                state = state + delta
                
        # Unpack
        state = state.squeeze(0).numpy()
        split = self.n * 3
        pos_final = state[:split].reshape(self.n, 3)
        vel_final = state[split:].reshape(self.n, 3)
        
        return pos_final, vel_final
