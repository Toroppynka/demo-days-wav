
import numpy as np

def f(audio: np.ndarray, mean: float = 0.0, sigma: float = 0.02) -> np.ndarray:
    """Добавляем Гауссов шум."""
    noise = np.random.normal(mean, sigma, size=audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)
