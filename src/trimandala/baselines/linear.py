
class LinearBaseline:
    """
    A dumb baseline that assumes Zero Acceleration.
    p(t) = p(0) + v(0) * t
    v(t) = v(0)
    """
    def predict(self, pos, vel, dt, steps):
        # pos shape (N, 3)
        # vel shape (N, 3)
        
        total_time = steps * dt
        
        pos_final = pos + vel * total_time
        vel_final = vel # Constant velocity
        
        return pos_final, vel_final
