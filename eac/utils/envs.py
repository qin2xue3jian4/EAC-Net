import os
import torch
import random

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
colors = [
    'blue','red','green','orange','yellow',
    'green','cyan','purple','pink','magenta','brown',
]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

def setup_seed(seed=0):
    seed = seed % (2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if os.environ.get('KEEP_SAME', '0') == '1':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return None
