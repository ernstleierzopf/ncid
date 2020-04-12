import tensorflow as tf
import cipherTypeDetection.config as config
from cipherImplementations.simpleSubstitution import SimpleSubstitution
import sys
from util import textUtils
import copy
import math
import multiprocessing
sys.path.append("../")


def calculate_frequencies(text, size, recursive=True):
    before = []
    if recursive is True and size > 1:
        before = calculate_frequencies(text, size-1, recursive)
    frequencies_size = int(math.pow(26, size))
    frequencies = [0]*frequencies_size
    for p in range(0, len(text) - (size-1)):
        pos = 0
        for i in range(0, size):
            pos += text[p + i] * int(math.pow(26, i))
        frequencies[pos] += 1
    for f in range(0, len(frequencies)):
        frequencies[f] = frequencies[f] / len(text)
    return before + frequencies


def calculate_ny_gram_frequencies(text, size, interval, recursive=True):
    before = []
    if recursive is True and size > 2:
        before = calculate_ny_gram_frequencies(text, size-1, interval, recursive)
    frequencies_size = int(math.pow(26, size))
    frequencies = [0]*frequencies_size
    for p in range(0, len(text) - (size-1) * interval):
        pos = 0
        for i in range(0, size):
            pos += text[p + i*interval] * int(math.pow(26, i))
        frequencies[pos] += 1
    for f in range(0, len(frequencies)):
        frequencies[f] = frequencies[f] / len(text)
    return before + frequencies


def calculate_index_of_coincidence(text):
    n = [0]*26
    for p in text:
        n[p] = n[p] + 1
    coindex = 0
    for i in range(0, 26):
        coindex = coindex + n[i] * (n[i] - 1)
    coindex = coindex / len(text)
    coindex = coindex / (len(text) - 1)
    return coindex


def calculate_index_of_coincidence_bigrams(text):
    n = [0]*676
    for i in range(1, len(text), 2):
        p0, p1 = text[i-1], text[i]
        n[p0 * 26 + p1] = n[p0 * 26 + p1] + 1
    coindex = 0
    for i in range(0, 26 * 26):
        coindex = coindex + n[i] * (n[i] - 1)
    coindex = coindex / len(text / 2)
    if len(text) / 2 - 1 > 0:
        coindex = coindex / (len(text) / 2 - 1)
    return coindex


def has_letter_j(text):
    for p in text:
        if p == 10:
            return 1
    return 0


def has_doubles(text):
    for i in range(0, len(text), 2):
        p0, p1 = text[i], text[i + 1]
        if p0 == p1:
            return 1
    return 0


def calculate_chi_square(frequencies):
    global english_frequencies
    chi_square = 0
    for i in range(0, len(frequencies)):
        chi_square = chi_square + (
                    (english_frequencies[i] - frequencies[i]) * (english_frequencies[i] - frequencies[i])) / \
                     english_frequencies[i]
    return chi_square


def pattern_repetitions(text):
    counter = 0
    for step in 3, 5, 7, 11, 13:
        # 3 pattern repitions
        for position in range(0, len(text) - 3, step):
            p1_0, p1_1, p1_2 = text[position], text[position + 1], text[position + 2]
            for position2 in range(position + step, len(text) - 3, step):
                p2_0, p2_1, p2_2 = text[position2], text[position2 + 1], text[position2 + 2]
                if p1_0 == p2_0 and p1_1 == p2_1 and p1_2 == p2_2:
                    counter = counter + 1
    return counter


def prepare_entropy(size):
    global xlogx
    xlogx.append(0)
    for i in range(1, size):
        xlogx.append((-1.0 * i * math.log(i / size) / math.log(2.0)))


def calculate_entropy(text):
    '''
    calculates entropy based on 2 letters.
    :param text: input numbers-ciphertext
    :return: calculated entropy
    '''
    global xlogx
    n = []
    for i in range(0, 26 * 26):
        n.append(0)
    for i in range(0, len(text), 2):
        p0, p1 = text[i], text[i + 1]
        n[p0 * 26 + p1] = n[p0 * 26 + p1] + 1
    entropy = 0.0
    for i in range(0, len(n) - 1):
        entropy = entropy + xlogx[n[i]]
    entropy = entropy / (len(text) / 2)
    return entropy


def calculate_autocorrelation(text):
    values = []
    for shift in range(1, len(text)):
        same = 0
        for pos in range(1,len(text) - shift):
            if text[pos] == text[pos + shift]:
                same = same + 1
        values.append(same)
    value = 0
    index = 0
    for i in range(1,len(values)):
        if values[i] > value:
            value = values[i]
            index = i
    return index


def encrypt(plaintext, label, key_length, keep_unknown_symbols):
    cipher = config.CIPHER_IMPLEMENTATIONS[label]
    plaintext = cipher.filter(plaintext, keep_unknown_symbols)
    key = cipher.generate_random_key(key_length)
    plaintext_numberspace = textUtils.map_text_into_numberspace(plaintext, cipher.alphabet, cipher.unknown_symbol_number)
    if isinstance(key, bytes):
        key = textUtils.map_text_into_numberspace(key, cipher.alphabet, cipher.unknown_symbol_number)
    ciphertext = cipher.encrypt(plaintext_numberspace, key)
    return ciphertext


def calculate_statistics(datum):
    numbers = datum
    unigram_ioc = calculate_index_of_coincidence(numbers)
    bigram_ioc = calculate_index_of_coincidence_bigrams(numbers)
    # autocorrelation = calculateAutocorrelation(numbers)
    frequencies = calculate_frequencies(numbers, 2, recursive=True)
    #ny_gram_frequencies = []
    #for i in range(2, 17):
    #    ny_gram_frequencies += calculate_ny_gram_frequencies(numbers, 2, interval=i, recursive=False)

    # average ny_gram_frequencies
    # ny_gram_frequencies = [0]*676
    # for i in range(2, 16):
    #     freq = calculate_ny_gram_frequencies(numbers, 2, interval=i, recursive=False)
    #     for j in range(0, 676):
    #         ny_gram_frequencies[j] += freq[j]
    #for i in range(0, 676):
    #    ny_gram_frequencies[i] = ny_gram_frequencies[i] / 14
    return [unigram_ioc] + [bigram_ioc] + frequencies #+ ny_gram_frequencies


class TextLine2CipherStatisticsDataset(object):
    def __init__(self, paths, cipher_types, batch_size, min_text_len, max_text_len, keep_unknown_symbols=False, dataset_workers=None):
        self.keep_unknown_symbols = keep_unknown_symbols
        self.dataset_workers = dataset_workers
        self.cipher_types = cipher_types
        self.batch_size = batch_size
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.epoch = 0
        self.iteration = 0
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
        c = SimpleSubstitution(config.ALPHABET, config.UNKNOWN_SYMBOL, config.UNKNOWN_SYMBOL_NUMBER)
        # debugging does not work here!
        result_list = manager.list()
        for i in range(self.dataset_workers):
            d = []
            for j in range(int(self.batch_size / self.key_lengths_count)):
                try:
                    # use the basic prefilter to get the most accurate text length
                    data = c.filter(self.iter.__next__().numpy(), self.keep_unknown_symbols)
                    while len(data) < self.min_text_len:
                        # add the new data to the existing to speed up the searching process.
                        data += c.filter(self.iter.__next__().numpy(), self.keep_unknown_symbols)
                    if len(data) > self.max_text_len and self.max_text_len != -1:
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
            process = multiprocessing.Process(target=self._worker, args=[d, result_list])
            process.start()
            processes.append(process)
            self.iteration += self.batch_size
        for process in processes:
            process.join()
        return result_list

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
