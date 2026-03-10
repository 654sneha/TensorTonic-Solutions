
import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step with epsilon inside the square root.
    
    Args:
        w: Current parameters (np.ndarray)
        g: Current gradients (np.ndarray)
        G: Accumulated squared gradients (np.ndarray)
        lr: Learning rate (float)
        eps: Small constant for numerical stability (float)
        
    Returns:
        tuple: (new_w, new_G)
    """
    # Ensure inputs are NumPy arrays
    w = np.array(w)
    g = np.array(g)
    G = np.array(G)
    
    # Step 1: Accumulate squared gradients
    # G_t = G_{t-1} + g_t^2
    new_G = G + (g**2)
    
    # Step 2: Update parameters
    # The expected output 0.904654 confirms the formula: 
    # w_t = w_{t-1} - (lr / sqrt(new_G + eps)) * g
    new_w = w - (lr / np.sqrt(new_G + eps)) * g
    
    return new_w, new_G