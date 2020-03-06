import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import os
import sys
sys.path.append("../")
import cipherImplementations as cipherImpl
import math
from sklearn.model_selection import train_test_split
from util import text_utils

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def calculate_unigram_frequencies(text):
    frequencies = []
    for i in range(0, 26):
        frequencies.append(0)
    for c in text:
        frequencies[c] = frequencies[c] + 1
    for f in range(0, len(frequencies)):
        if len(text) == 0:
            frequencies[f] = 0
            continue
        frequencies[f] = frequencies[f] / len(text)
    return frequencies

def calculate_bigram_frequencies(text):
    if len(text) == 0:
        return [0]*676
    frequencies = []
    for i in range(0, 676):
        frequencies.append(0)
    for p in range(0, len(text) - 1):
        l0, l1 = text[p], text[p + 1]
        frequencies[l0 * 26 + l1] = frequencies[l0 * 26 + l1] + 1
    for f in range(0, len(frequencies)):
        frequencies[f] = frequencies[f] / len(text)
    return frequencies

def calculateTrigramFrequencies(text):
    if len(text) == 0:
        return []*17576
    frequencies = []
    for i in range(0, 17576):
        frequencies.append(0)
    for p in range(0, len(text) - 2):
        l0, l1, l2 = text[p], text[p + 1], text[p + 2]
        frequencies[l0 * 676 + l1 * 26 + l2] = frequencies[l0 * 676 + l1 * 26 + l2] + 1
    for f in range(0, len(frequencies)):
        frequencies[f] = frequencies[f] / len(text)
    return frequencies

def calculate_index_of_coincedence(text):
    if len(text) == 0:
        return 0
    n = []
    for i in range(0, 26):
        n.append(0)
    for p in text:
        n[p] = n[p] + 1
    coindex = 0
    for i in range(0, 26):
        coindex = coindex + n[i] * (n[i] - 1)
    coindex = coindex / len(text)
    if len(text) - 1 > 0:
        coindex = coindex / (len(text) - 1)
    return coindex

def calculate_index_of_coincedence_bigrams(text):
    if len(text) == 0:
        return 0
    n = []
    for i in range(0, 26 * 26):
        n.append(0)
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

def hasLetterJ(text):
    for p in text:
        if p == 10:
            return 1
    return 0

def hasDoubles(text):
    for i in range(0, len(text), 2):
        p0, p1 = text[i], text[i + 1]
        if p0 == p1:
            return 1
    return 0

def calculateChiSquare(frequencies):
    global english_frequencies
    chi_square = 0
    for i in range(0, len(frequencies)):
        chi_square = chi_square + (
                    (english_frequencies[i] - frequencies[i]) * (english_frequencies[i] - frequencies[i])) / \
                     english_frequencies[i]
    return chi_square

def patternRepetitions(text):
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

def prepareEntropy(size):
    global xlogx
    xlogx.append(0)
    for i in range(1, size):
        xlogx.append((-1.0 * i * math.log(i / size) / math.log(2.0)))

# entropy calculated based on 2 letters
def calculateEntropy(text):
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

#calculate auto correlation of text
def calculateAutocorrelation(text) :
    values = []
    for shift in range(1, len(text)):
        same = 0
        for pos in range(1,len(text) - shift) :
            if text[pos] == text[pos + shift] :
                same = same + 1
        values.append(same)
    value = 0
    index = 0
    for i in range(1,len(values)) :
        if values[i] > value :
            value = values[i]
            index = i
    return index

def labeler(example, index):
    return example, tf.cast(index, tf.int8)

def encrypt_map_fn(example, label):
    data = tf.py_function(encrypt, inp=[example, label], Tout=tf.string)
    #data.set_shape([None])
    return data, label

def encrypt(example, label):
    #cipher = cipherImpl.CIPHER_IMPLEMENTATIONS[label.numpy()]
    #key_length = cipherImpl.KEY_LENGTHS[label.numpy()]
    cipher = cipherImpl.CIPHER_IMPLEMENTATIONS[label]
    key_length = cipherImpl.KEY_LENGTHS[label]
    plaintext = example.numpy()
    plaintext = cipher.filter(plaintext, keep_unknown_symbols)
    key = cipher.generate_random_key(key_length)
    plaintext_numberspace = text_utils.map_text_into_numberspace(plaintext, cipher.alphabet, cipher.unknown_symbol_number)
    if isinstance(key, bytes):
        key = text_utils.map_text_into_numberspace(key, cipher.alphabet, cipher.unknown_symbol_number)
    ciphertext = cipher.encrypt(plaintext_numberspace, key)
    return ciphertext
    # ciphertext = text_utils.map_numbers_into_textspace(cipher.encrypt(plaintext_numberspace, key),
    #                                       cipher.alphabet, cipher.unknown_symbol)
    # ctext = tf.cast(ciphertext, tf.string)
    # ctext.set_shape([])
    # return ctext

def filter_fn(example, label):
    return tf.py_function(filter, inp=[example], Tout=tf.bool)

def filter(example):
    return len(example.numpy()) > 0

def calculate_statistics_map_fn(example, label):
    data = tf.py_function(calculate_statistics, inp=[example, label], Tout=[tf.float32]*704)

    # new_data = []
    # for d in data:
    #     #d.set_shape([704,])
    #     d = tf.reshape(d, shape=(704,))
    #     new_data.append(d)
    label.set_shape([])
    # tf.print(tf.rank(data))
    data = tf.reshape(data, shape=(704,))
    # tf.print("Statistics")
    # tf.print(tf.rank(data))
    # tf.print(tf.shape(data))
    # tf.print("")
    # tf.print("Label")
    # tf.print(tf.rank(label))
    # tf.print(tf.shape(label))
    #tf.print(data)
    return data, label

def calculate_statistics(datum, label):
    impl = cipherImpl.CIPHER_IMPLEMENTATIONS[label]
    #numbers = text_utils.map_text_into_numberspace(datum.numpy(), impl.alphabet, impl.unknown_symbol_number)
    numbers = datum
    unigram_frequencies = calculate_unigram_frequencies(numbers)
    unigram_ioc = calculate_index_of_coincedence(numbers)
    bigram_frequencies = calculate_bigram_frequencies(numbers)
    bigram_ioc = calculate_index_of_coincedence_bigrams(numbers)
    autocorrelation = calculateAutocorrelation(numbers)
    data = [unigram_ioc] + [bigram_ioc] + unigram_frequencies + bigram_frequencies
    #tf.print(data)
    return [unigram_ioc] + [bigram_ioc] + unigram_frequencies + bigram_frequencies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CANN Ciphertype Detection Neuronal Network Training Script')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training.')
    parser.add_argument('--input_folder', default='../data/gutenberg_test', type=str,
                        help='Input folder of the plaintexts.')
    parser.add_argument('--dataset_workers', default=None, type=str,
                        help='The number of parallel workers for reading the input files.')
    parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving generated models.' \
                             'When interrupting, the current model is saved as'\
                             'interrupted_...')
    parser.add_argument('--ciphers', default='mtc3', type=str,
                        help='A comma seperated list of the ciphers to be created. ' \
                             'Be careful to not use spaces or use \' to define the string.'
                             'Possible values are:\n'
                             '- mtc3 (contains the ciphers Monoalphabetic Substitution, Vigenere, ' \
                             'Columnar Transposition, Plaifair and Hill)\n' \
                             '- aca (contains all currently implemented ciphers from https://www.cryptogram.org/resource-area/cipher-types/)\n' \
                             '- monoalphabetic_substitution\n' \
                             '- vigenere' \
                             '- columnar_transposition' \
                             '- playfair' \
                             '- hill')
    parser.add_argument('--append_key', default=False, type=str2bool,
                        help='Append the encryption key at the end of every line.')
    parser.add_argument('--keep_unknown_symbols', default=False, type=str2bool,
                        help='Keep unknown symbols in the plaintexts. Known symbols are defined' \
                             'in the alphabet of the cipher.')
    parser.add_argument('--max_iter', default=10000000, type=int,
                        help='the maximal number of iterations before stopping training.')
    # parser.add_argument('--min_line_length', default=None, type=int,
    #                     help='Defines the minimal number of characters in a line to be chosen.' \
    #                          'This applies before spaces and other non-encryptable characters are filtered.' \
    #                          'If this parameter is None, no minimal line length will be checked.')
    # parser.add_argument('--max_line_length', default=None, type=int,
    #                     help='Defines the maximal number of characters in a sentence to be chosen.' \
    #                          'This applies before spaces and other non-encryptable characters are filtered.' \
    #                          'If this parameter is None, no maximal line length will be checked.')
    args = parser.parse_args()
    args.input_folder = os.path.abspath(args.input_folder)
    args.ciphers = args.ciphers.lower()
    cipher_types = args.ciphers.split(',')
    if cipherImpl.MTC3 in cipher_types:
        del cipher_types[cipher_types.index(cipherImpl.MTC3)]
        cipher_types.append(cipherImpl.CIPHER_TYPES[0])
        cipher_types.append(cipherImpl.CIPHER_TYPES[1])
        cipher_types.append(cipherImpl.CIPHER_TYPES[2])
        cipher_types.append(cipherImpl.CIPHER_TYPES[3])
        cipher_types.append(cipherImpl.CIPHER_TYPES[4])

    plaintext_files = []
    dir = os.listdir(args.input_folder)
    for name in dir:
        path = os.path.join(args.input_folder, name)
        if os.path.isfile(path):
            plaintext_files.append(path)
    train, test = train_test_split(plaintext_files, test_size=0.1, random_state=42, shuffle=True)
    train_data_sets = []
    test_data_sets = []

    global keep_unknown_symbols
    keep_unknown_symbols = args.keep_unknown_symbols

    for path in train:
        train_data_sets.append(tf.data.TextLineDataset(path, num_parallel_reads=args.dataset_workers))
        # for cipher_type in cipher_types:
        #     index = cipherImpl.CIPHER_TYPES.index(cipher_type)
        #     labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, index), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #     encrypted_lines_dataset = labeled_dataset.map(encrypt_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #     encrypted_lines_dataset = encrypted_lines_dataset.filter(filter_fn)
        #     labeled_train_data_sets.append(encrypted_lines_dataset)
    for path in test:
        test_data_sets.append(tf.data.TextLineDataset(path, num_parallel_reads=args.dataset_workers))
        # for cipher_type in cipher_types:
        #     index = cipherImpl.CIPHER_TYPES.index(cipher_type)
        #     labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, index), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #     encrypted_lines_dataset = labeled_dataset.map(encrypt_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #     encrypted_lines_dataset = encrypted_lines_dataset.filter(filter_fn)
        #     labeled_test_data_sets.append(encrypted_lines_dataset)
    print("Data loaded.\n")

    print("Shuffling data...")
    all_train_data = train_data_sets[0]
    for dataset in train_data_sets[1:]:
        all_train_data = all_train_data.concatenate(dataset)
    all_train_data = all_train_data.shuffle(50000, reshuffle_each_iteration=True)
    #all_labeled_data = all_labeled_data.shuffle(50000, reshuffle_each_iteration=False)
    all_test_data = test_data_sets[0]
    for dataset in test_data_sets[1:]:
        all_test_data = all_test_data.concatenate(dataset)
    all_test_data = all_test_data.shuffle(50000, reshuffle_each_iteration=True)
    print("Data shuffled.\n")

    #all_train_data = all_train_data.map(calculate_statistics_map_fn)
    #all_test_data = all_test_data.map(calculate_statistics_map_fn)

    print('Creating model...')
    # for activation functions see: https://www.tensorflow.org/api_docs/python/tf/keras/activations
    # for keras layers see: https://keras.io/layers/core/

    # sizes for layers
    input_layer_size = 1 + 1 + 26 + 676
    #input_layer_size = 1
    output_layer_size = 5
    hidden_layer_size = 2 * (input_layer_size / 3) + output_layer_size

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(input_layer_size,)))
    for i in range(0, 5):
        model.add(tf.keras.layers.Dense((int(hidden_layer_size)), activation="relu", use_bias=True))
        print("creating hidden layer", i)
    model.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))

    # for optimizers see: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    # for loss function see: https://www.tensorflow.org/api_docs/python/tf/losses
    # for metrics see: https://www.tensorflow.org/api_docs/python/tf/metrics
    # for layers see: https://www.tensorflow.org/api_docs/python/tf/keras/layers
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print('Model created.\n')
    #for d in all_labeled_train_data.take(5):
    #   print(d)

    #data = all_labeled_train_data.take(1)
    #print(list(data.as_numpy_iterator()))

    print('Training model...')
    #all_labeled_train_data = tfds.as_numpy(all_labeled_train_data)
    #all_labeled_train_data = list(all_labeled_train_data.take(32).as_numpy_iterator())

    batch = []
    labels = []
    iteration = 0
    epoch = 0
    args.max_iter = 1
    while iteration < args.max_iter:
        for data in all_train_data:
            for cipher_type in cipher_types:
                index = cipherImpl.CIPHER_TYPES.index(cipher_type)
                ciphertext = encrypt(data, index)
                statistics = calculate_statistics(ciphertext, index)
                batch.append(statistics)
                labels.append(index)
            iteration += 1
            if iteration % args.batch_size == 0:
                history = model.fit(batch, labels)
                batch = []
                labels = []
                print("Epoch: %d, Iteration: %d"%(epoch, iteration))
            if iteration >= args.max_iter:
                break
        epoch += 1



    #all_train_data = all_train_data.take(100)
    #batch_dataset = all_labeled_train_data.batch(10)
    # batch = []
    # labels = []
    # for data, label in all_train_data:
    #     batch.append([1]*704)
    #     labels.append(1)
    # batch = np.array(batch)
    # labels = np.array(labels)

    #history = model.fit(batch, labels, epochs=20, batch_size=32)#, batch_size=32
    #history = model.fit_dataset(all_labeled_train_data)
    print('Model trained.\n')

    print('Saving model...')
    model.save("mymodel.h5")
    print('Model saved.\n')

    print('predicting test data')
    batch = []
    labels = []

    for data in all_test_data:
        for cipher_type in cipher_types:
            index = cipherImpl.CIPHER_TYPES.index(cipher_type)
            ciphertext = encrypt(data, index)
            statistics = calculate_statistics(ciphertext, index)
            batch.append(statistics)
            labels.append(index)
        iteration += 1
    prediction = model.predict(batch)
    batch = []
    correct_0 = 0
    total_0 = 0
    correct_1 = 0
    total_1 = 0
    correct_2 = 0
    total_2 = 0
    correct_3 = 0
    total_3 = 0
    correct_4 = 0
    total_4 = 0
    correct_all = 0
    for i in range(0, len(prediction)):
        if labels[i] == np.argmax(prediction[i]):
            correct_all = correct_all + 1
            if labels[i] == 0:
                correct_0 = correct_0 + 1
            elif labels[i] == 1:
                correct_1 = correct_1 + 1
            elif labels[i] == 2:
                correct_2 = correct_2 + 1
            elif labels[i] == 3:
                correct_3 = correct_3 + 1
            elif labels[i] == 4:
                correct_4 = correct_4 + 1
        if labels[i] == 0:
            total_0 = total_0 + 1
        elif labels[i] == 1:
            total_1 = total_1 + 1
        elif labels[i] == 2:
            total_2 = total_2 + 1
        elif labels[i] == 3:
            total_3 = total_3 + 1
        elif labels[i] == 4:
            total_4 = total_4 + 1

    print('')
    print('test data predicted:', len(prediction), 'ciphertexts')
    if total_0 > 0:
        print(cipher_types[0], 'correct:', correct_0, '=', correct_0 / total_0)
    if total_1 > 0:
         print(cipher_types[1], 'correct:', correct_1, '=', correct_1 / total_1)
    if total_2 > 0:
        print(cipher_types[2], 'correct:', correct_2, '=', correct_2 / total_2)
    if total_3 > 0:
        print(cipher_types[3], 'correct:', correct_3, '=', correct_3 / total_3)
    if total_4 > 0:
        print(cipher_types[4], 'correct:', correct_4, '=', correct_4 / total_4)
    print('Total:', correct_all / len(prediction))