import random
import numpy as np
import torch
import os
import shutil


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)


def init_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)
    os.mkdir(os.path.join(path, 'log'))
    os.mkdir(os.path.join(path, 'run'))
    os.mkdir(os.path.join(path, 'weights'))
    os.mkdir(os.path.join(path, 'images'))
