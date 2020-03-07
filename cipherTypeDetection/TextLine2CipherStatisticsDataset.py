import tensorflow as tf
import tensorflow_datasets.public_api as tfds


class TextLine2CipherStatisticsDataset:
    def __init__(self, paths, keep_unknown_symbols=False, dataset_workers=None, download_dataset=True):
        self.keep_unknown_symbols = keep_unknown_symbols
        self.dataset_workers = dataset_workers
        self.download_dataset = download_dataset

        datasets = []

        for path in paths:
            datasets.append(tf.data.TextLineDataset(path, num_parallel_reads=dataset_workers))
        for path in datasets:
            datasets.append(tf.data.TextLineDataset(path, num_parallel_reads=dataset_workers))

        self.dataset = datasets[0]
        for dataset in datasets[1:]:
            self.dataset = self.dataset.concatenate(dataset)

    def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
        return self.dataset.shuffle(buffer_size, seed, reshuffle_each_iteration)
