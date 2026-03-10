import numpy as np

def adam_step(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # CRITICAL: Convert all inputs to numpy arrays
    # This prevents the "can't multiply sequence by non-int" error
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)

    # 1. Update biased first moment estimate
    # Now that 'm' and 'grad' are arrays, this math works element-wise
    m_new = beta1 * m + (1 - beta1) * grad
    
    # 2. Update biased second raw moment estimate
    v_new = beta2 * v + (1 - beta2) * (grad**2)
    
    # 3. Bias correction
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)
    
    # 4. Update parameters
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return param_new, m_new, v_new