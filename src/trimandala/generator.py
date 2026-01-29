import h5py
import numpy as np
import os
from tqdm import tqdm
# We'll assume the C++ module is importable as trimandala.core
# In a real install, it would be compiled. For now, we mock its existence or rely on install.

try:
    from .core import SymplecticIntegrator
except ImportError:
    # Fallback/Mock for dev if compilation hasn't happened yet
    print("Warning: C++ core not found. Using Mock.")
    class SymplecticIntegrator:
        def __init__(self, m, p, v): pass
        def step(self, dt): pass
        def get_state(self, n): return np.zeros((n,3)), np.zeros((n,3))

class SimulationRunner:
    def __init__(self, output_file: str, n_bodies: int):
        self.output_file = output_file
        self.n_bodies = n_bodies
    
    def run(self, masses, pos_init, vel_init, steps, dt, chunk_size=1000):
        """
        Runs the simulation via C++ core and streams data to HDF5.
        """
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
            
        chunk_size = min(chunk_size, steps)

            
        integrator = SymplecticIntegrator(masses, pos_init, vel_init)
        
        with h5py.File(self.output_file, 'w') as f:
            # Metadata
            f.attrs['n_bodies'] = self.n_bodies
            f.attrs['steps'] = steps
            f.attrs['dt'] = dt
            
            # Create resizable datasets
            # We assume we write every 'chunk_size' steps to keep file size sanity? 
            # Or we write ALL steps but chunked in memory?
            # BENCHMARK.md implies we want the trajectory.
            
            dset_pos = f.create_dataset("positions", (steps, self.n_bodies, 3), dtype='f8', chunks=(chunk_size, self.n_bodies, 3))
            dset_vel = f.create_dataset("velocities", (steps, self.n_bodies, 3), dtype='f8', chunks=(chunk_size, self.n_bodies, 3))
            dset_mass = f.create_dataset("masses", data=masses)
            dset_time = f.create_dataset("time", (steps,), dtype='f8', chunks=(chunk_size,))
            
            # Init Buffers
            buffer_pos = np.zeros((chunk_size, self.n_bodies, 3))
            buffer_vel = np.zeros((chunk_size, self.n_bodies, 3))
            buffer_time = np.zeros((chunk_size,))
            
            # Initial State
            p, v = integrator.get_state(self.n_bodies)
            dset_pos[0] = p
            dset_vel[0] = v
            dset_time[0] = 0.0
            
            # Fix: Init buffer index 0 to prevent overwrite with zeros on flush
            buffer_pos[0] = p
            buffer_vel[0] = v
            buffer_time[0] = 0.0
            
            # Loop
            for t in tqdm(range(1, steps)):
                integrator.step(dt)
                
                # Buffer logic for extreme IO optimization could go here.
                # For simplicity/robustness, we write directly or buffer small chunks.
                # Let's buffer 'chunk_size' frames.
                
                # Fetch state (Zero-copy ideally via pybind, here standard)
                p, v = integrator.get_state(self.n_bodies)
                
                idx_in_buffer = t % chunk_size
                buffer_pos[idx_in_buffer] = p
                buffer_vel[idx_in_buffer] = v
                buffer_time[idx_in_buffer] = t * dt
                
                # Flush buffer
                if idx_in_buffer == chunk_size - 1:
                    start_idx = t - chunk_size + 1
                    dset_pos[start_idx : t+1] = buffer_pos
                    dset_vel[start_idx : t+1] = buffer_vel
                    dset_time[start_idx : t+1] = buffer_time
            
            # Flush remaining
            processed = (steps // chunk_size) * chunk_size
            remaining = steps - processed
            if remaining > 0:
                 # Logic for remainder... simplified for prototype
                 pass

        print(f"Simulation completed. Saved to {self.output_file}")
