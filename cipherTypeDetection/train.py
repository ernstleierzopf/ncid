from pathlib import Path

import numpy as np
import argparse
import sys
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
# This environ variable must be set before all tensorflow imports!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
sys.path.append("../")
import cipherImplementations as cipherImpl
from cipherTypeDetection.textLine2CipherStatisticsDataset import TextLine2CipherStatisticsDataset
tf.debugging.set_log_device_placement(enabled=False)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CANN Ciphertype Detection Neuronal Network Training Script')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for training.')
    parser.add_argument('--train_dataset_size', default=16000, type=int,
                        help='Dataset size per fit. This argument should be dividable by the amount of --ciphers.')
    parser.add_argument('--dataset_workers', default=1, type=int,
                        help='The number of parallel workers for reading the input files.')
    parser.add_argument('--input_folder', default='../data/gutenberg_en', type=str,
                        help='Input folder of the plaintexts.')
    parser.add_argument('--download_dataset', default=True, type=str2bool,
                        help='Download the dataset automatically.')
    parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving generated models.' \
                             'When interrupting, the current model is saved as'\
                             'interrupted_...')
    parser.add_argument('--model_name', default='model.h5', type=str,
                        help='Name of the output model file. The file must have the .h5 extension.')
    parser.add_argument('--ciphers', default='mtc3', type=str,
                        help='A comma seperated list of the ciphers to be created. ' \
                             'Be careful to not use spaces or use \' to define the string.'
                             'Possible values are:\n'
                             '- mtc3 (contains the ciphers Monoalphabetic Substitution, Vigenere, ' \
                             'Columnar Transposition, Plaifair and Hill)\n' \
                             '- aca (contains all currently implemented ciphers from https://www.cryptogram.org/resource-area/cipher-types/)\n' \
                             '- simple_substitution\n' \
                             '- vigenere' \
                             '- columnar_transposition' \
                             '- playfair' \
                             '- hill')
    parser.add_argument('--keep_unknown_symbols', default=False, type=str2bool,
                        help='Keep unknown symbols in the plaintexts. Known symbols are defined' \
                             'in the alphabet of the cipher.')
    parser.add_argument('--max_iter', default=1000000, type=int,
                        help='the maximal number of iterations before stopping training.')
    args = parser.parse_args()
    m = os.path.splitext(args.model_name)
    if len(os.path.splitext(args.model_name)) != 2 or os.path.splitext(args.model_name)[1] != '.h5':
        print('ERROR: The model name must have the ".h5" extension!', file=sys.stderr)
        exit(1)
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
    if args.train_dataset_size % len(cipher_types) != 0:
        print("WARNING: the --batch_size parameter must be dividable by the amount of --ciphers.", file=sys.stderr)
    if args.train_dataset_size * args.dataset_workers > args.max_iter:
        print("ERROR: --train_dataset_size * --dataset_workers must not be bigger than --max_iter. "
              "In this case it was %d > %d" % (args.train_dataset_size * args.dataset_workers, args.max_iter), file=sys.stderr)
        exit(1)

    if args.download_dataset and not os.path.exists(args.input_folder) and args.input_folder == os.path.abspath('../data/gutenberg_en'):
        print("Downloading Datsets...")
        tfds.download.add_checksums_dir('../data/checksums/')
        download_manager = tfds.download.download_manager.DownloadManager(download_dir='../data/', extract_dir=args.input_folder)
        download_manager.download_and_extract('https://drive.google.com/uc?id=1bF5sSVjxTxa3DB-P5wxn87nxWndRhK_V&export=download')
        path = os.path.join(args.input_folder, 'ZIP.ucid_1bF5sSVjxTx-P5wxn87nxWn_V_export_downloadR9Cwhunev5CvJ-ic__HawxhTtGOlSdcCrro4fxfEI8A.incomplete_25fe7c1666cb4a8fb06682d99df2c0df', os.path.basename(args.input_folder))
        dir = os.listdir(path)
        for name in dir:
            p = Path(os.path.join(path, name))
            parent_dir = p.parents[2]
            p.rename(parent_dir / p.name)
        os.rmdir(path)
        os.rmdir(os.path.dirname(path))
        print("Datasets Downloaded.")

    print("Loading Datasets...")
    plaintext_files = []
    dir = os.listdir(args.input_folder)
    # dir = ['2.txt','3.txt','4.txt','5.txt']
    for name in dir:
        path = os.path.join(args.input_folder, name)
        if os.path.isfile(path):
            plaintext_files.append(path)
    train, test = train_test_split(plaintext_files, test_size=0.1, random_state=42, shuffle=True)

    train_dataset = TextLine2CipherStatisticsDataset(train, cipher_types, args.train_dataset_size, args.keep_unknown_symbols, args.dataset_workers)
    test_dataset = TextLine2CipherStatisticsDataset(test, cipher_types, args.train_dataset_size, args.keep_unknown_symbols, args.dataset_workers)
    print("Datasets loaded.\n")

    print("Shuffling data...")
    train_dataset = train_dataset.shuffle(50000, seed=42, reshuffle_each_iteration=True)
    test_dataset = test_dataset.shuffle(50000, seed=42, reshuffle_each_iteration=True)
    print("Data shuffled.\n")

    print('Creating model...')
    # for activation functions see: https://www.tensorflow.org/api_docs/python/tf/keras/activations
    # for keras layers see: https://keras.io/layers/core/

    # for optimizers see: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    # for loss function see: https://www.tensorflow.org/api_docs/python/tf/losses
    # for metrics see: https://www.tensorflow.org/api_docs/python/tf/metrics
    # for layers see: https://www.tensorflow.org/api_docs/python/tf/keras/layers

    # sizes for layers
    input_layer_size = 1 + 1 + 26 + 676
    output_layer_size = 5
    hidden_layer_size = 2 * (input_layer_size / 3) + output_layer_size


    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Flatten(input_shape=(input_layer_size,)))
    # for i in range(0, 5):
    #     model.add(tf.keras.layers.Dense((int(hidden_layer_size)), activation="relu", use_bias=True))
    #     print("creating hidden layer", i)
    # model.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
    # model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # logistic regression baseline
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    #     model = tf.keras.Sequential()
    #     model.add(tf.keras.layers.Dense(output_layer_size, input_dim=input_layer_size, activation='softmax', use_bias=True))
    #     model.compile(optimizer='sgd', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(output_layer_size, input_dim=input_layer_size, activation='softmax', use_bias=True))
    model.compile(optimizer='sgd', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print('Model created.\n')

    print('Training model...')
    import time
    start_time = time.time()
    cntr = 0
    while train_dataset.iteration < args.max_iter:
        for run in train_dataset:
            for batch, labels in run:
                if train_dataset.iteration >= args.max_iter:
                    break
                history = model.fit(batch, labels, batch_size=args.batch_size, workers=args.dataset_workers)
                cntr += 1
                iteration = args.train_dataset_size * cntr
                epoch = iteration // (train_dataset.iteration / train_dataset.epoch)
                print("Epoch: %d, Iteration: %d" % (epoch, iteration))
            if train_dataset.iteration >= args.max_iter:
                break
    elapsed_training_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)
    print('Finished training in %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.\n' %
           (elapsed_training_time.days, elapsed_training_time.seconds // 3600, (elapsed_training_time.seconds // 60) % 60,
           (elapsed_training_time.seconds) % 60, train_dataset.iteration, train_dataset.epoch))

    print('Saving model...')
    model.save(os.path.join(args.save_folder, args.model_name))
    print('Model saved.\n')

    print('Predicting test data...')
    start_time = time.time()
    correct = [0]*len(cipherImpl.CIPHER_TYPES)
    total = [0]*len(cipherImpl.CIPHER_TYPES)
    correct_all = 0
    total_len_prediction = 0
    while test_dataset.iteration < args.max_iter / 10:
        for run in test_dataset:
            for batch, labels in run:
                if test_dataset.iteration >= args.max_iter / 10:
                    break
                prediction = model.predict(batch, batch_size=args.batch_size, workers=args.dataset_workers)
                for i in range(0, len(prediction)):
                    if labels[i] == np.argmax(prediction[i]):
                        correct_all += 1
                        correct[labels[i]] += 1
                    total[labels[i]] += 1
                total_len_prediction += len(prediction)
                print("Prediction Epoch: %d, Iteration: %d / %d" % (test_dataset.epoch, test_dataset.iteration, args.max_iter / 10))
            if test_dataset.iteration >= args.max_iter / 10:
                break
    elapsed_prediction_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)

    total_len_prediction -= total_len_prediction%args.train_dataset_size
    print('\ntest data predicted: %d ciphertexts'%total_len_prediction)
    for i in range(0, len(total)):
        if total[i] == 0:
            continue
        print('%s correct: %d/%d = %f'%(cipherImpl.CIPHER_TYPES[i], correct[i], total[i], correct[i] / total[i]))
    if total_len_prediction == 0:
        t = 'N/A'
    else:
        t = str(correct_all / total_len_prediction)
    print('Total: %s\n'%t)
    print('Training time: %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.' %(
          elapsed_training_time.days, elapsed_training_time.seconds // 3600, (elapsed_training_time.seconds // 60) % 60,
          (elapsed_training_time.seconds) % 60, train_dataset.iteration, train_dataset.epoch))
    print('Prediction time: %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.' %(
          elapsed_prediction_time.days, elapsed_prediction_time.seconds // 3600, (elapsed_prediction_time.seconds // 60) % 60,
          (elapsed_prediction_time.seconds) % 60, test_dataset.iteration, test_dataset.epoch))
