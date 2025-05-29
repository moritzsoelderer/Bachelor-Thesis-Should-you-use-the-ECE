import numpy as np

"""
    This code was adapted in parts from:
    
    **Calibration of Neural Networks using Splines**  
    GitHub Repository: https://github.com/kartikgupta-at-anu/spline-calibration
    
    Disclaimer:
"""

def ksce(scores, labels):
    scores = ensure_numpy(scores)
    labels = ensure_numpy(labels)

    scores = scores[:, 1]

    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]

    n_samples = scores.shape[0]
    integrated_scores = np.cumsum(scores) / n_samples
    integrated_accuracy = np.cumsum(labels) / n_samples

    return np.amax(np.absolute(integrated_scores - integrated_accuracy))


def ensure_numpy(a):
    if not isinstance(a, np.ndarray): a = a.numpy()
    return a