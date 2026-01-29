import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def plot_energy_drift(h5_file: str, output_path: str = "energy_drift.png"):
    """
    Plots Relative Energy Error vs Time.
    """
    from trimandala.metrics.rigor import _hamiltonian
    
    with h5py.File(h5_file, 'r') as f:
        t = f['time'][:]
        m = f['masses'][:]
        pos = f['positions']
        vel = f['velocities']
        
        # Sample every 100th step to save memory plotting
        indices = np.arange(0, len(t), 100)
        
        energies = []
        for i in indices:
            energies.append(_hamiltonian(m, pos[i], vel[i]))
        
        energies = np.array(energies)
        drift = np.abs((energies - energies[0]) / energies[0])
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(t[indices], drift, label='Energy Drift')
        plt.xlabel('Time')
        plt.ylabel('|(E(t)-E0)/E0|')
        plt.title('Symplectic Integrator Stability')
        plt.grid(True, which="both", ls="-")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved energy plot to {output_path}")

def plot_phase_space(h5_file: str, body_idx: int = 0, output_path: str = "phase_space.png"):
    """
    Plots velocity vs position (x vs vx) for a specific body.
    """
    with h5py.File(h5_file, 'r') as f:
        p = f['positions'][:, body_idx, 0] # X
        v = f['velocities'][:, body_idx, 0] # VX
        
        plt.figure(figsize=(8, 8))
        plt.plot(p, v, lw=0.5, alpha=0.7)
        plt.xlabel('Position X')
        plt.ylabel('Velocity X')
        plt.title(f'Phase Space Projection (Body {body_idx})')
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

def export_to_alembic(h5_file: str, output_abc: str):
    """
    Exports simulation to Alembic (.abc) format for Blender.
    Requires 'alembic' python library or similar. 
    For V1, we will just output a generic OBJ sequence or simplified format 
    as proper Alembic binding is heavy.
    
    Placeholder: Writes a Blender Script to import the HDF5 data.
    """
    script_content = f"""
import bpy
import h5py
import numpy as np

file_path = "{h5_file}"

def load_traj():
    with h5py.File(file_path, 'r') as f:
        pos = f['positions'][:] # Warning: Load all? Streaming needed for Blender?
        # For huge files, this script needs to be smarter.
        # But for 1GB ram, loading 200MB is fine.
        
    # Create spheres
    for i in range(pos.shape[1]):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1)
        obj = bpy.context.active_object
        obj.name = f"Body_{{i}}"
        
        # Animate
        for t in range(0, pos.shape[0], 10): # Skip frames for speed
            obj.location = pos[t, i, :]
            obj.keyframe_insert(data_path="location", frame=t/10)

load_traj()
    """
    with open(output_abc + ".py", "w") as f:
        f.write(script_content)
    print(f"Generated Blender import script: {output_abc}.py")
