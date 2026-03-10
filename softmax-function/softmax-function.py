
import numpy as np

def softmax(x):
    """
    Computes the numerically stable Softmax of a 1D or 2D array.
    
    Arguments:
    x -- A NumPy array (or list) of any shape.
    
    Returns:
    A NumPy array of the same shape where elements sum to 1 
    along the last axis.
    """
    x = np.array(x)
    
    # 1. Numerical Stability: Find the max value along the last axis
    # keepdims=True ensures we can subtract it from 'x' correctly (broadcasting)
    max_x = np.max(x, axis=-1, keepdims=True)
    
    # 2. Exponentiate the shifted values
    e_x = np.exp(x - max_x)
    
    # 3. Normalize by the sum of exponents along the last axis
    sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
    
    return e_x / sum_e_x

# --- Testing the examples ---

# 1D Input
input_1d = np.array([1, 2, 3])
print(f"1D Softmax: {softmax(input_1d)}")
# Expected: [0.09003057, 0.24472847, 0.66524096]

# 2D Input (Matrix)
input_2d = np.array([[1, 2, 3], [0, 0, 0]])
print(f"\n2D Softmax:\n{softmax(input_2d)}")
# Row 1 expected: [0.09, 0.24, 0.66]
# Row 2 expected: [0.33, 0.33, 0.33]