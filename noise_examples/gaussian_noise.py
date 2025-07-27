
"""Additive Gaussian noise for a mono float32 numpy audio array in [-1, 1]."""

import numpy as np

def add_gaussian_noise(audio: np.ndarray, mean: float = 0.0, sigma: float = 0.01) -> np.ndarray:
    """Return a copy of *audio* with Gaussian noise added."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    noise = np.random.normal(mean, sigma, size=audio.shape).astype(np.float32)
    noisy = np.clip(audio + noise, -1.0, 1.0)
    return noisy
