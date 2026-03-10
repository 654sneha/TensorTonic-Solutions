import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    
    Args:
        w: Current parameters (np.ndarray)
        g: Current gradients (np.ndarray)
        s: Running squared gradient accumulator (np.ndarray)
        lr: Learning rate (float)
        beta: Decay factor (float)
        eps: Small constant for numerical stability (float)
        
    Returns:
        tuple: (updated_w, updated_s)
    """
    # Ensure inputs are numpy arrays to allow element-wise math
    w = np.array(w)
    g = np.array(g)
    s = np.array(s)

    # Step 1: Update the running average of squared gradients
    # s_t = beta * s_{t-1} + (1 - beta) * g_t^2
    new_s = beta * s + (1 - beta) * (g**2)
    
    # Step 2: Update the parameters
    # w_t = w_{t-1} - (lr / sqrt(s_t + eps)) * g_t
    new_w = w - (lr / (np.sqrt(new_s) + eps)) * g
    
    return new_w, new_s