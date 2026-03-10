import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    """
    Works for:
    a, b → (d,) OR (N, d)
    y → scalar OR (N,)
    """

    a = np.array(a)
    b = np.array(b)
    y = np.array(y)

    # Ensure 2D shape (N, d)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    if y.ndim == 0:
        y = y.reshape(1)

    # Euclidean distance per pair
    d = np.linalg.norm(a - b, axis=1)

    # Contrastive loss per sample
    loss = y * (d ** 2) + (1 - y) * (np.maximum(0, margin - d) ** 2)

    # Reduction
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError("Invalid reduction type")


# ---------------------------------------
# Example 1 (single pair)
# ---------------------------------------
a = [1, 2, 3]
b = [1, 2, 4]
y = 1
print("Single pair loss:", contrastive_loss(a, b, y))

# ---------------------------------------
# Example 2 (batch)
# ---------------------------------------
a = [[1, 2], [3, 4], [5, 6]]
b = [[1, 2.1], [2.5, 4], [8, 9]]
y = [1, 0, 0]
print("Batch loss:", contrastive_loss(a, b, y))