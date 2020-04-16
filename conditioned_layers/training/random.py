import numpy as np
import torch
import random

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fraction_dataset(targets, classes, fraction):
    indices = []
    for klass in classes:
        class_indices = [i for i, e in enumerate(targets) if e == klass]
        subset_size = math.floor(len(class_indices) * fraction)
        indices.extend(random.sample(class_indices, subset_size))

    random.shuffle(indices)
    return indices