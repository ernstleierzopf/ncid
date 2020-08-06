import tensorflow as tf
import cipherTypeDetection.config as config
from cipherImplementations.cipher import OUTPUT_ALPHABET
from cipherImplementations.simpleSubstitution import SimpleSubstitution
import sys
from util.textUtils import map_text_into_numberspace
import copy
import math
import multiprocessing
sys.path.append("../")
import numpy as np


english_frequencies = [
    0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749, 0.07507,
    0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758, 0.00978, 0.0236, 0.0015, 0.01974, 0.00074]


def calculate_frequencies(text, size, recursive=True):
    before = []
    if recursive is True and size > 1:
        before = calculate_frequencies(text, size-1, recursive)
    frequencies_size = int(math.pow(len(OUTPUT_ALPHABET), size))
    frequencies = [0]*frequencies_size
    for p in range(0, len(text) - (size-1)):
        pos = 0
        for i in range(0, size):
            pos += text[p + i] * int(math.pow(len(OUTPUT_ALPHABET), i))
        frequencies[pos] += 1
    for f in range(0, len(frequencies)):
        frequencies[f] = frequencies[f] / len(text)
    return before + frequencies


def calculate_ny_gram_frequencies(text, size, interval, recursive=True):
    before = []
    if recursive is True and size > 2:
        before = calculate_ny_gram_frequencies(text, size-1, interval, recursive)
    frequencies_size = int(math.pow(len(OUTPUT_ALPHABET), size))
    frequencies = [0]*frequencies_size
    for p in range(0, len(text) - (size-1) * interval):
        pos = 0
        for i in range(0, size):
            pos += text[p + i*interval] * int(math.pow(len(OUTPUT_ALPHABET), i))
        frequencies[pos] += 1
    for f in range(0, len(frequencies)):
        frequencies[f] = frequencies[f] / len(text)
    return before + frequencies


def calculate_index_of_coincidence(text):
    n = [0]*len(OUTPUT_ALPHABET)
    for p in text:
        n[p] = n[p] + 1
    coindex = 0
    for i in range(0, len(OUTPUT_ALPHABET)):
        coindex = coindex + n[i] * (n[i] - 1)
    coindex = coindex / len(text)
    coindex = coindex / (len(text) - 1)
    return coindex


def calculate_index_of_coincidence_bigrams(text):
    n = [0]*(len(OUTPUT_ALPHABET) * len(OUTPUT_ALPHABET))
    for i in range(1, len(text), 1):
        p0, p1 = text[i-1], text[i]
        n[p0 * len(OUTPUT_ALPHABET) + p1] = n[p0 * len(OUTPUT_ALPHABET) + p1] + 1
    coindex = 0
    for i in range(0, len(OUTPUT_ALPHABET) * len(OUTPUT_ALPHABET)):
        coindex += n[i] * (n[i] - 1)
    coindex = coindex / len(text) / (len(text) - 1) / (len(text) - 2)
    return coindex


def has_letter_j(text):
    for p in text:
        if p == 9:
            return 1
    return 0


def has_doubles(text):
    for i in range(0, len(text), 2):
        p0, p1 = text[i], text[i + 1]
        if p0 == p1:
            return 1
    return 0


def calculate_chi_square(frequencies):
    chi_square = 0
    for i in range(0, len(frequencies)):
        chi_square = chi_square + (
                    (english_frequencies[i] - frequencies[i]) * (english_frequencies[i] - frequencies[i])) / \
                     english_frequencies[i]
    return chi_square


def pattern_repetitions(text):
    counter = 0
    patterns = []
    for i in range(0, len(text) - 2):
        pattern = [text[i], text[i + 1], text[i + 2]]
        if pattern not in patterns:
            patterns.append(pattern)
            for j in range(i + 1, len(text) - 2):
                if pattern == [text[j], text[j + 1], text[j + 2]]:
                    counter += 1
    return counter


def calculate_entropy(text):
    """calculates shannon's entropy index.
    :param text: input numbers-ciphertext
    :return: calculated entropy"""
    # https://stackoverflow.com/questions/2979174/how-do-i-compute-the-approximate-entropy-of-a-bit-string
    _unique, counts = np.unique(text, return_counts=True)
    prob = []
    for c in counts:
        prob.append(float(c) / len(text))
    # calculate the entropy
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy


def calculate_autocorrelation_average(text):
    """calculates average of the normalized autocorrelation
    :param text: input numbers-ciphertext
    :return: autocorrelation average"""
    # https://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python
    n = len(text)
    variance = text.var()
    text = text - text.mean()
    r = np.correlate(text, text, mode='full')[-n:]
    result = r / (variance * (np.arange(n, 0, -1)))
    avg = 0
    for i in range(0, len(result)):
        avg += result[i]
    avg = avg / len(result)
    return avg


def encrypt(plaintext, label, key_length, keep_unknown_symbols):
    cipher = config.CIPHER_IMPLEMENTATIONS[label]
    plaintext = cipher.filter(plaintext, keep_unknown_symbols)
    key = cipher.generate_random_key(key_length)
    plaintext_numberspace = map_text_into_numberspace(plaintext, cipher.alphabet, cipher.unknown_symbol_number)
    if isinstance(key, bytes):
        key = map_text_into_numberspace(key, cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 2 and isinstance(key[0], bytes) and isinstance(key[1], bytes):
        key[0] = map_text_into_numberspace(key[0], cipher.alphabet, cipher.unknown_symbol_number)
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 3 and isinstance(key[0], bytes) and isinstance(key[1], bytes) and isinstance(key[2], bytes):
        key[0] = map_text_into_numberspace(key[0], cipher.alphabet, cipher.unknown_symbol_number)
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
        key[2] = map_text_into_numberspace(key[2], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 2 and isinstance(key[0], bytes) and isinstance(key[1], int):
        key[0] = map_text_into_numberspace(key[0], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 3 and isinstance(key[0], int) and isinstance(key[1], bytes) and isinstance(key[2], bytes):
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
        key[2] = map_text_into_numberspace(key[2], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 2 and (isinstance(key[0], list) or isinstance(key[0], np.ndarray)) and (
            len(key[0]) == 5 or len(key[0]) == 10) and isinstance(key[1], bytes):
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 3 and isinstance(key[0], list) and isinstance(key[1], np.ndarray) and isinstance(
            key[2], bytes):
        key[2] = map_text_into_numberspace(key[2], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, dict):
        new_key_dict = {}
        for k in key:
            new_key_dict[cipher.alphabet.index(k)] = key[k]
        key = new_key_dict

    ciphertext = cipher.encrypt(plaintext_numberspace, key)
    if b'j' not in cipher.alphabet and config.CIPHER_TYPES[label] != 'homophonic':
        ciphertext = normalize_text(ciphertext, 9)
    if b'x' not in cipher.alphabet:
        ciphertext = normalize_text(ciphertext, 23)
    return ciphertext


def normalize_text(text, pos):
    for i in range(len(text)):
        if text[i] >= pos:
            text[i] += 1
    return text


def calculate_statistics(datum):
    numbers = datum
    unigram_ioc = calculate_index_of_coincidence(numbers)
    bigram_ioc = calculate_index_of_coincidence_bigrams(numbers)
    # autocorrelation = calculate_autocorrelation_average(numbers)
    frequencies = calculate_frequencies(numbers, 2, recursive=True)
    # has_j = has_letter_j(numbers)
    # has_double = has_doubles(numbers)
    # chi_square = calculate_chi_square(frequencies[0:26])
    # pattern_repetitions_count = pattern_repetitions(numbers)
    # entropy = calculate_entropy(numbers)

    # ny_gram_frequencies = []
    # for i in range(2, 16):
    #     ny_gram_frequencies += calculate_ny_gram_frequencies(numbers, 2, interval=i, recursive=False)

    # average ny_gram_frequencies
    # ny_gram_frequencies = [0]*676
    # for i in range(2, 16):
    #     freq = calculate_ny_gram_frequencies(numbers, 2, interval=i, recursive=False)
    #     for j in range(0, 676):
    #         ny_gram_frequencies[j] += freq[j]
    # for i in range(0, 676):
    #    ny_gram_frequencies[i] = ny_gram_frequencies[i] / 14

    # ny_gram_frequencies = []
    # ny_gram_frequencies += calculate_ny_gram_frequencies(numbers, 2, interval=5, recursive=False)
    # ny_gram_frequencies += calculate_ny_gram_frequencies(numbers, 2, interval=10, recursive=False)
    # ny_gram_frequencies += calculate_ny_gram_frequencies(numbers, 2, interval=20, recursive=False)
    # ny_gram_frequencies += calculate_ny_gram_frequencies(numbers, 2, interval=25, recursive=False)
    return [unigram_ioc] + [bigram_ioc] + frequencies  # + ny_gram_frequencies


class TextLine2CipherStatisticsDataset:
    def __init__(self, paths, cipher_types, batch_size, min_text_len, max_text_len, keep_unknown_symbols=False, dataset_workers=None):
        self.keep_unknown_symbols = keep_unknown_symbols
        self.dataset_workers = dataset_workers
        self.cipher_types = cipher_types
        self.batch_size = batch_size
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.epoch = 0
        self.iteration = 0
        self.iter = None
        datasets = []
        for path in paths:
            datasets.append(tf.data.TextLineDataset(path, num_parallel_reads=dataset_workers))
        self.dataset = datasets[0]
        for dataset in datasets[1:]:
            self.dataset = self.dataset.zip(dataset)
        count = 0
        for cipher_t in self.cipher_types:
            index = config.CIPHER_TYPES.index(cipher_t)
            if isinstance(config.KEY_LENGTHS[index], list):
                count += len(config.KEY_LENGTHS[index])
            else:
                count += 1
        self.key_lengths_count = count

    def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
        new_dataset = copy.copy(self)
        new_dataset.dataset = new_dataset.dataset.shuffle(buffer_size, seed, reshuffle_each_iteration)
        return new_dataset

    def __iter__(self):
        self.iter = self.dataset.__iter__()
        return self

    def __next__(self):
        processes = []
        manager = multiprocessing.Manager()
        c = SimpleSubstitution(config.INPUT_ALPHABET, config.UNKNOWN_SYMBOL, config.UNKNOWN_SYMBOL_NUMBER)
        # debugging does not work here!
        result_list = manager.list()
        for _ in range(self.dataset_workers):
            d = []
            for _ in range(int(self.batch_size / self.key_lengths_count)):
                try:
                    # use the basic prefilter to get the most accurate text length
                    data = c.filter(self.iter.__next__().numpy(), self.keep_unknown_symbols)
                    while len(data) < self.min_text_len:
                        # add the new data to the existing to speed up the searching process.
                        data += c.filter(self.iter.__next__().numpy(), self.keep_unknown_symbols)
                    if len(data) > self.max_text_len != -1:
                        d.append(data[:self.max_text_len])
                    else:
                        d.append(data)
                except:
                    self.epoch += 1
                    self.__iter__()
                    data = c.filter(self.iter.__next__().numpy(), self.keep_unknown_symbols)
                    while len(data) < self.min_text_len:
                        data += c.filter(self.iter.__next__().numpy(), self.keep_unknown_symbols)
                    if len(data) > self.max_text_len:
                        d.append(data[:self.max_text_len])
                    else:
                        d.append(data)
            process = multiprocessing.Process(target=self._worker, args=(d, result_list))
            process.start()
            processes.append(process)
        return processes, result_list

    def _worker(self, data, result):
        batch = []
        labels = []
        for d in data:
            for cipher_t in self.cipher_types:
                index = config.CIPHER_TYPES.index(cipher_t)
                if isinstance(config.KEY_LENGTHS[index], list):
                    key_lengths = config.KEY_LENGTHS[index]
                else:
                    key_lengths = [config.KEY_LENGTHS[index]]
                for key_length in key_lengths:
                    ciphertext = encrypt(d, index, key_length, self.keep_unknown_symbols)
                    statistics = calculate_statistics(ciphertext)
                    batch.append(statistics)
                    labels.append(index)
        result.append((tf.convert_to_tensor(batch), tf.convert_to_tensor(labels)))