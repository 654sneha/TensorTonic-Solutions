import numpy as np

def relu(x):
    """
    Applies the Rectified Linear Unit activation function.
    
    Arguments:
    x -- Input (scalar, list, or NumPy array)
    
    Returns:
    A NumPy array where negative values are replaced by 0.
    """
    # Convert input to a numpy array to handle lists or scalars
    x = np.array(x)
    
    # np.maximum compares each element in x against 0 and picks the larger one
    return np.maximum(0, x)

# --- Testing the examples ---
print(f"List Input: {relu([-2, -1, 0, 3])}") 
# Output: [0. 0. 0. 3.]

print(f"Scalar Input: {relu(5.0)}")           
# Output: 5.0

print(f"Matrix Input:\n{relu([[-1, 2], [3, -4]])}") 
# Output: [[0. 2.], [3. 0.]]




