# utils/seed.py

import random
import numpy as np
import torch

def set_random_seed(seed: int = 42):
    """
    Set random seed for Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed number to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
