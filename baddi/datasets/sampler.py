import torch.utils.data
import logging
import numpy as np
from torch.utils.data.sampler import RandomSampler, BatchSampler
from typing import Callable
import pandas as pd


class SamplingConfig:
    def __init__(self, balance=False, class_weights=None):
        if balance and class_weights is not None:
            raise ValueError("Params 'balance' and 'weights' are incompatible")
        self._balance = balance
        self._class_weights = class_weights

    def create_loader(self, dataset, batch_size):
        if self._balance:
            sampler = BalancedBatchSampler(dataset.targets, batch_size=batch_size)
        elif self._class_weights is not None:
            sampler = ReweightedBatchSampler(
                dataset.targets,
                batch_size=batch_size,
                class_weights=self._class_weights,
            )
        else:
            sampler = BatchSampler(
                ImbalancedDatasetSampler(labels=dataset.targets),
                batch_size=batch_size,
                drop_last=False,
            )

        return torch.utils.data.DataLoader(dataset=dataset, batch_sampler=sampler)


class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_samples for each of the n_classes.
    Returns batches of size n_classes * (batch_size // n_classes)
    adapted from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, dataset, batch_size, drop_last: bool = False):
        labels = dataset
        classes = np.unique(labels)

        n_classes = len(classes)
        self._n_samples = batch_size // n_classes
        if self._n_samples == 0:
            raise ValueError(
                f"batch_size should be bigger than the number of classes, got {batch_size}"
            )

        self._class_iters = [
            InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_)
            for class_ in classes
        ]

        batch_size = self._n_samples * n_classes
        self.n_dataset = len(labels)
        self._n_batches = self.n_dataset // batch_size
        if self._n_batches == 0:
            raise ValueError(
                f"Dataset is not big enough to generate batches with size {batch_size}"
            )
        self.batch_size = batch_size
        self.drop_last = drop_last

        print("K=", n_classes, "nk=", self._n_samples)
        print("Batch size = ", batch_size)

    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            for class_iter in self._class_iters:
                indices.extend(class_iter.get(self._n_samples))
            np.random.shuffle(indices)
            yield indices

        for class_iter in self._class_iters:
            class_iter.reset()

    def __len__(self):
        return self._n_batches


class ReweightedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples batch_size according to given input distribution
    assuming multi-class labels
    adapted from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    # /!\ 'class_weights' should be provided in the "natural order" of the classes (i.e. sorted(classes)) /!\
    def __init__(self, dataset, batch_size, class_weights, drop_last: bool = False):
        labels = dataset
        self._classes = np.unique(labels)

        n_classes = len(self._classes)
        if n_classes > len(class_weights):
            k = len(class_weights)
            sum_w = np.sum(class_weights)
            if sum_w >= 1:
                # normalize attributing equal weight to weighted part and remaining part
                class_weights /= sum_w * k / n_classes + (n_classes - k) / n_classes
            krem = k - n_classes
            wrem = 1 - sum_w
            logging.warning(
                f"will assume uniform distribution for labels > {len(class_weights)}"
            )
            self._class_weights = np.ones(n_classes, dtype=np.float)
            self._class_weights[:k] = class_weights
            self._class_weights[k:] = wrem / krem
        else:
            self._class_weights = class_weights[:n_classes]

        if np.sum(self._class_weights) != 1:
            self._class_weights = self._class_weights / np.sum(self._class_weights)

        print("Using weights=", self._class_weights)
        if batch_size == 0:
            raise ValueError(
                f"batch_size should be bigger than the number of classes, got {batch_size}"
            )

        self._class_to_iter = {
            class_: InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_)
            for class_ in self._classes
        }

        self.n_dataset = len(labels)
        self._batch_size = batch_size
        self._n_batches = self.n_dataset // self._batch_size
        if self._n_batches == 0:
            raise ValueError(
                f"Dataset is not big enough to generate batches with size {self._batch_size}"
            )
        print("K=", n_classes, "nk=", self._batch_size)
        print("Batch size = ", self._batch_size)

        self.batch_size = self._batch_size
        self.drop_last = drop_last

    def __iter__(self):
        for _ in range(self._n_batches):
            # sample batch_size classes
            class_idx = np.random.choice(
                self._classes,
                p=self._class_weights,
                replace=True,
                size=self._batch_size,
            )
            indices = []
            for class_, num in zip(*np.unique(class_idx, return_counts=True)):
                indices.extend(self._class_to_iter[class_].get(num))
            np.random.shuffle(indices)
            yield indices

        for class_iter in self._class_to_iter.values():
            class_iter.reset()

    def __len__(self):
        return self._n_batches


class InfiniteSliceIterator:
    def __init__(self, array, class_):
        assert type(array) is np.ndarray
        self.array = array
        self.i = 0
        self.class_ = class_

    def reset(self):
        self.i = 0

    def get(self, n):
        len_ = len(self.array)
        # not enough element in 'array'
        if len_ < n:
            logging.debug(f"there are really few items in class {self.class_}")
            self.reset()
            np.random.shuffle(self.array)
            mul = n // len_
            rest = n - mul * len_
            return np.concatenate((np.tile(self.array, mul), self.array[:rest]))

        # not enough element in array's tail
        if len_ - self.i < n:
            self.reset()

        if self.i == 0:
            np.random.shuffle(self.array)
        i = self.i
        self.i += n
        return self.array[i : self.i]


# Look like weighted random rampler
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset"""

    def __init__(
        self,
        labels: list = None,
    ):
        # all elements in the dataset will be considered
        self.indices = list(range(len(labels)))

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples
