import numpy as np

def pca_rgb(arr_hw_d: np.ndarray, eps: float = 1e-12):
    """
    arr_hw_d: array of shape (H, W, D) with arbitrary scale (not normalized)
    Returns: (H, W, 3) uint8 RGB image of the first 3 PCs
    """
    H, W, D = arr_hw_d.shape
    X = arr_hw_d.reshape(-1, D).astype(np.float64)      # (N, D), N = H*W

    # Center (mean subtraction) — required for PCA
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    # PCA via SVD of the centered data
    # Xc = U S Vt, rows=N samples, cols=D features
    # Principal directions are rows of Vt; scores = Xc @ Vt.T
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ Vt.T                                   # (N, D)

    # Take first 3 components (pad with zeros if D<3)
    K = min(3, D)
    pcs3 = np.zeros((X.shape[0], 3), dtype=np.float64)
    pcs3[:, :K] = scores[:, :K]

    # Scale each channel to 0..1 for display (min-max per channel)
    ch_min = pcs3.min(axis=0, keepdims=True)
    ch_max = pcs3.max(axis=0, keepdims=True)
    denom = np.maximum(ch_max - ch_min, eps)
    pcs3_norm = (pcs3 - ch_min) / denom                  # (N, 3) in [0,1]

    rgb = (pcs3_norm.reshape(H, W, 3) * 255).astype(np.uint8)
    return rgb
