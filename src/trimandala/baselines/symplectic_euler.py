
def symplectic_euler_step(pos, vel, dt, masses):
    """
    First-order Symplectic Euler (Semi-implicit).
    v(t+1) = v(t) + a(x(t)) * dt
    x(t+1) = x(t) + v(t+1) * dt
    """
    # 1. Compute Forces (N-Body O(N^2))
    # This is the bottleneck for Python
    import numpy as np
    
    n = len(masses)
    acc = np.zeros_like(pos)
    
    # Vectorized compute is better than double loops in Python
    # But for N=3, loops are fine. Let's do vectorized for "Baseline" quality.
    
    # Broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3) diff matrix
    # diff[i, j] = pos[i] - pos[j]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] # dx
    
    dist_sq = np.sum(diff**2, axis=-1) + 1e-10 # (N, N) r^2
    dist = np.sqrt(dist_sq) # (N, N) r
    dist_cube = dist_sq * dist # r^3
    
    # F_ij = G * mi * mj / r^3 * r_vec
    # a_i = Sum(F_ij / mi) = Sum(G * mj / r^3 * r_vec)
    # G=1
    
    # accel contribution matrix
    # We want to sum over j.
    # We need to mask diagonal (i=j) to avoid division by zero self-force
    # But we added softening 1e-10, so it's small? No, 1/epsilon is huge.
    # Cleaner to mask.
    
    inv_r3 = 1.0 / dist_cube
    np.fill_diagonal(inv_r3, 0.0) # Self-interaction is zero
    
    # acc[i] = Sum_j ( mass[j] * inv_r3[i, j] * (pos[j] - pos[i]) )
    # Note diff[i, j] is pos[i] - pos[j] = -r_vec
    # So Force ~ -diff
    
    # We need Sum_j ( m[j] * inv_r3[i,j] * (-diff[i,j]) )
    
    # (N, N) masses broadcast
    m_j = masses[np.newaxis, :]
    
    # Scalar factor per pair
    scalar = m_j * inv_r3 # (N, N)
    
    # Compute acc: sum over j (axis 1)
    # scalar is (N, N), diff is (N, N, 3)
    # We want (N, 3)
    
    # Force is attractive. Vector pointing to j is (pos[j] - pos[i]) = -diff[i,j]
    acc = np.sum(scalar[:, :, np.newaxis] * (-diff), axis=1)
    
    # 2. Update
    vel_next = vel + acc * dt
    pos_next = pos + vel_next * dt
    
    return pos_next, vel_next
