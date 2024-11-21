"""
Put functions here that wouldn't fit anywhere else but are too small to make their own files
"""

import logging
import os
import random
from contextlib import contextmanager
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import Any, Dict, Generator, Optional

import torch

def seed_everything(seed, cuda_deterministic=True):
    
    random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True   
    return seed