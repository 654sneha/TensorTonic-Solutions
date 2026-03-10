
import numpy as np

def conv2d(x, W, b):
    # Ensure inputs are treated as numpy arrays for shape and math operations
    x = np.array(x)
    W = np.array(W)
    b = np.array(b)
    
    # Get dimensions (N: Batch, C: Channels, H: Height, W: Width)
    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape
    
    # Calculate output dimensions for 'valid' convolution
    H_out = H - KH + 1
    W_out = W_in - KW + 1
    
    # Initialize output array
    y = np.zeros((N, C_out, H_out, W_out))
    
    # Ensure b is indexable even if passed as a 0-d array
    if b.ndim == 0:
        b = np.array([b])
        
    # Iterate through the output spatial dimensions
    for i in range(H_out):
        for j in range(W_out):
            # Slicing: (N, C_in, KH, KW)
            x_patch = x[:, :, i : i + KH, j : j + KW]
            
            # Vectorized sum over C_in, KH, and KW for each output channel
            for cout in range(C_out):
                # Multiply patch by the specific filter and sum
                # np.sum with axis=(1,2,3) keeps the batch dimension (N)
                y[:, cout, i, j] = np.sum(x_patch * W[cout], axis=(1, 2, 3)) + b[cout]
                
    return y