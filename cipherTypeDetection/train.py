import tensorflow as tf
import argparse
import os
import sys
import cipherImplementations as cipherImpl
import math
from sklearn.model_selection import train_test_split

sys.path.append("../")
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
        frequencies[f] = frequencies[f] / len(text)
    return frequencies

def calculate_bigram_frequencies(text):
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

def calculate_statistics(datum):
    impl = cipherImpl.CIPHER_IMPLEMENTATIONS[index]
    numbers = text_utils.map_text_into_numberspace(datum, impl.alphabet, impl.unknown_symbol_number)
    unigram_frequencies = calculate_unigram_frequencies(numbers)
    unigram_ioc = calculate_index_of_coincedence(numbers)
    bigram_frequencies = calculate_bigram_frequencies(numbers)
    bigram_ioc = calculate_index_of_coincedence_bigrams(numbers)
    autocorrelation = calculateAutocorrelation(numbers)
    # texts.append([unigram_ioc] +
    #             [bigram_ioc] +
    #             # [autocorrelation] +
    #             unigram_frequencies +
    #             bigram_frequencies)
    return [unigram_ioc], [bigram_ioc], [autocorrelation], unigram_frequencies, bigram_frequencies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CANN Ciphertype Detection Neuronal Network Training Script')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size for training.')
    parser.add_argument('--ciphertexts_base_dir', default='../data/mtc3_cipher_id', type=str,
                        help='Base directory of the ciphertexts.')
    parser.add_argument('--dataset_workers', default=4, type=str,
                        help='The number of parallel workers for reading the input files.')
    parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving generated models.' \
                             'When interrupting, the current model is saved as'\
                             'interrupted_...')
    args = parser.parse_args()
    args.ciphertexts_base_dir = os.path.abspath(args.ciphertexts_base_dir)

    sub_dirs = []
    dir = os.listdir(args.ciphertexts_base_dir)
    for name in dir:
        path = os.path.join(args.ciphertexts_base_dir, name)
        if os.path.isdir(path):
            sub_dirs.append(path)

    labeled_train_data_sets = []
    labeled_test_data_sets = []
    for sub_dir in sub_dirs:
        print("Loading %s..."%sub_dir)
        dir = os.listdir(sub_dir)
        train, test = train_test_split(dir, test_size=0.1, random_state=42, shuffle=True)
        if os.path.basename(sub_dir) in cipherImpl.CIPHER_TYPES:
            index = cipherImpl.CIPHER_TYPES.index(os.path.basename(sub_dir))
            for name in train:
                path = os.path.join(sub_dir, name)
                if os.path.isfile(path):
                    lines_dataset = tf.data.TextLineDataset(path, num_parallel_reads=args.dataset_workers)
                    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, index))
                    labeled_train_data_sets.append(labeled_dataset)
            for name in test:
                path = os.path.join(sub_dir, name)
                if os.path.isfile(path):
                    lines_dataset = tf.data.TextLineDataset(path, num_parallel_reads=args.dataset_workers)
                    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, index))
                    labeled_test_data_sets.append(labeled_dataset)
        else:
            print("error: %s not found"%sub_dir)
            continue
    print("Data loaded.\n")

    print("Shuffling data...")
    all_labeled_train_data = labeled_train_data_sets[0]
    for labeled_dataset in labeled_train_data_sets[1:]:
        all_labeled_train_data = all_labeled_train_data.concatenate(labeled_dataset)
    all_labeled_train_data = all_labeled_train_data.shuffle(10000, reshuffle_each_iteration=False)
    #all_labeled_data = all_labeled_data.shuffle(50000, reshuffle_each_iteration=False)
    all_labeled_test_data = labeled_test_data_sets[0]
    for labeled_dataset in labeled_test_data_sets[1:]:
        all_labeled_test_data = all_labeled_test_data.concatenate(labeled_dataset)
    all_labeled_test_data = all_labeled_test_data.shuffle(10000, reshuffle_each_iteration=False)
    print("Data shuffled.\n")

    #l = list(all_labeled_data.as_numpy_iterator())
    for ex in all_labeled_train_data.take(5):
        print(ex[0].numpy())

    print('Creating model...')
    # for activation functions see: https://www.tensorflow.org/api_docs/python/tf/keras/activations
    # for keras layers see: https://keras.io/layers/core/

    # sizes for layers
    input_layer_size = 1 + 1 + 26 + 676
    # input_layer_size = 100
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

    print('Training model...')
    counter = 0
    for ex in all_labeled_train_data.take(1):
        train_dataset = calculate_statistics(ex[0].numpy())
    for datum in all_labeled_train_data:
        tensors = calculate_statistics(datum[0].numpy())
        train_dataset = zip(train_dataset, tf.data.Dataset.from_tensors(tensors))
        if counter % 10000 == 0:
            print(counter)
        counter += 1
    counter = 0
    for ex in all_labeled_test_data.take(1):
        test_dataset = calculate_statistics(ex[0].numpy())
    for datum in all_labeled_test_data:
        tensors = calculate_statistics(datum[0].numpy())
        test_dataset = zip(test_dataset, tf.data.Dataset.from_tensors(tensors))
        if counter % 10000 == 0:
            print(counter)
        counter += 1

    history = model.fit(train_dataset, validation_data=test_dataset, epochs=20, batch_size=32)
    print('Model trained.\n')

    print('Saving model...')
    model.save("mymodel.h5")
    print('Model saved.\n')

    print('predicting test data')
    # prediction = model.predict(test_texts)
    #
    # correct_0 = 0
    # total_0 = 0
    # correct_1 = 0
    # total_1 = 0
    # correct_2 = 0
    # total_2 = 0
    # correct_3 = 0
    # total_3 = 0
    # correct_4 = 0
    # total_4 = 0
    # correct_all = 0
    #
    # for i in range(0, len(prediction)):
    #     if test_labels[i] == np.argmax(prediction[i]):
    #         correct_all = correct_all + 1
    #         if test_labels[i] == 0:
    #             correct_0 = correct_0 + 1
    #         elif test_labels[i] == 1:
    #             correct_1 = correct_1 + 1
    #         elif test_labels[i] == 2:
    #             correct_2 = correct_2 + 1
    #         elif test_labels[i] == 3:
    #             correct_3 = correct_3 + 1
    #         elif test_labels[i] == 4:
    #             correct_4 = correct_4 + 1
    #     if test_labels[i] == 0:
    #         total_0 = total_0 + 1
    #     elif test_labels[i] == 1:
    #         total_1 = total_1 + 1
    #     elif test_labels[i] == 2:
    #         total_2 = total_2 + 1
    #     elif test_labels[i] == 3:
    #         total_3 = total_3 + 1
    #     elif test_labels[i] == 4:
    #         total_4 = total_4 + 1
    #
    # print('')
    # print('test data predicted:', len(prediction), 'ciphertexts')
    # if total_0 > 0:
    #     print(label_mapping_names[0], 'correct:', correct_0, '=', correct_0 / total_0)
    # if total_1 > 0:
    #     print(label_mapping_names[1], 'correct:', correct_1, '=', correct_1 / total_1)
    # if total_2 > 0:
    #     print(label_mapping_names[2], 'correct:', correct_2, '=', correct_2 / total_2)
    # if total_3 > 0:
    #     print(label_mapping_names[3], 'correct:', correct_3, '=', correct_3 / total_3)
    # if total_4 > 0:
    #     print(label_mapping_names[4], 'correct:', correct_4, '=', correct_4 / total_4)
    # print('Total:', correct_all / len(prediction))