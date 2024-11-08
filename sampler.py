# sampler.py
import torch
from torch.utils.data import Sampler


class NonObsoleteSampler(Sampler):
    def __init__(self, dataset, length=1000):
        """
        Initialize the sampler with a dataset. This sampler filters the dataset
        to only include indices where `id` is not 0 and `obsolete` is 0.

        Args:
            dataset (Dataset): The dataset to sample from.
        """
        self.dataset = dataset
        # 获取所有符合条件的有效索引：id 不为 0，且 obsolete 为 0
        self.valid_indices = [
            i for i in range(1, length+1)
            if dataset.get_obsolete(i) == 0
        ]

    def __iter__(self):
        """
        Return an iterator that yields shuffled indices from valid_indices.
        """
        indices = torch.randperm(len(self.valid_indices)).tolist()  # 随机打乱索引顺序
        return (self.valid_indices[i] for i in indices)

    def __len__(self):
        """
        Return the number of valid samples.
        """
        return len(self.valid_indices)
