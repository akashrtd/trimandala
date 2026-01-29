import torch
import torch.nn as nn
import numpy as np

class SimpleMLP(nn.Module):
    """
    A simple residual MLP that predicts the change in state.
    Delta = Model(State_t)
    State_t+1 = State_t + Delta
    """
    def __init__(self, n_bodies=3, hidden_size=512):
        super().__init__()
        self.n = n_bodies
        input_dim = n_bodies * 6 # 3 pos + 3 vel per body
        output_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, list(nn.modules.linear.Linear(1, 1).parameters())[0].shape[0] if False else hidden_size), # dynamic check? No, fixed.
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
        
    def forward(self, x):
        # x shape: (Batch, N*6)
        return self.net(x)

    def predict(self, pos, vel, dt, steps):
        """
        API for Arena.
        Iteratively rolls out the model.
        """
        # Convert to tensor
        # pos shape (N, 3), vel shape (N, 3) -> Flatten (1, N*6)
        
        # Ensure input is float32 for torch
        pos_t = torch.tensor(pos, dtype=torch.float32).flatten().unsqueeze(0)
        vel_t = torch.tensor(vel, dtype=torch.float32).flatten().unsqueeze(0)
        state = torch.cat([pos_t, vel_t], dim=1) # This is wrong flattening order if we want pos, vel interleaved? 
        # Let's standardize state vector: [p1x, p1y, p1z, ..., vn_z] vs [all_pos, all_vel]
        # HDF5 is [pos, vel]. Let's stick to [all_pos, all_vel] as simple concat.
        
        self.eval()
        with torch.no_grad():
            for _ in range(steps):
                # Predict delta
                delta = self.forward(state)
                state = state + delta
                
        # Unpack
        state = state.squeeze(0).numpy()
        split = self.n * 3
        pos_final = state[:split].reshape(self.n, 3)
        vel_final = state[split:].reshape(self.n, 3)
        
        return pos_final, vel_final
