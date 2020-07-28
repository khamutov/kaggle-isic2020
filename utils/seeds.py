import os
import random
import numpy as np
import torch


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(seed_value)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(seed_value)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = True