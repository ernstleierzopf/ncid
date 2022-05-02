from pathlib import Path

import numpy as np
import argparse
import sys
import time
import shutil
from sklearn.model_selection import train_test_split
import os
import math
import pickle
import functools
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from datetime import datetime
# This environ variable must be set before all tensorflow imports!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam  # , Adamax
import tensorflow_datasets as tfds
sys.path.append("../")
import cipherTypeDetection.config as config
from cipherImplementations.cipher import OUTPUT_ALPHABET
from cipherTypeDetection.textLine2CipherStatisticsDataset import TextLine2CipherStatisticsDataset
from cipherTypeDetection.miniBatchEarlyStoppingCallback import MiniBatchEarlyStopping
from cipherTypeDetection.transformer import TransformerBlock, TokenAndPositionEmbedding
from cipherTypeDetection.learningRateSchedulers import TimeBasedDecayLearningRateScheduler, CustomStepDecayLearningRateScheduler
tf.debugging.set_log_device_placement(enabled=False)
# always flush after print as some architectures like RF need very long time before printing anything.
print = functools.partial(print, flush=True)
# for device in tf.config.list_physical_devices('GPU'):
#    tf.config.experimental.set_memory_growth(device, True)


architecture = None


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def create_model():
    global architecture
    optimizer = Adam(
        learning_rate=config.learning_rate, beta_1=config.beta_1, beta_2=config.beta_2, epsilon=config.epsilon, amsgrad=config.amsgrad)
    # optimizer = Adamax()
    model_ = None

    # sizes for layers
    total_frequencies_size = 0
    for j in range(1, 3):
        total_frequencies_size += math.pow(len(OUTPUT_ALPHABET), j)
    total_frequencies_size = int(total_frequencies_size)

    # total_ny_gram_frequencies_size = int(math.pow(len(OUTPUT_ALPHABET), 2)) * 6

    input_layer_size = 18 + total_frequencies_size
    output_layer_size = len(cipher_types)
    hidden_layer_size = int(2 * (input_layer_size / 3) + output_layer_size)

    # logistic regression baseline
    # model_ = tf.keras.Sequential()
    # model_.add(tf.keras.layers.Dense(output_layer_size, input_dim=input_layer_size, activation='softmax', use_bias=True))
    # model_.compile(optimizer='sgd', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # extend model
    if extend_model is not None:
        # remove the last layer
        model_ = tf.keras.Sequential()
        for layer in extend_model.layers[:-1]:
            model_.add(layer)
        model_.add(tf.keras.layers.Dense(output_layer_size, activation='softmax', name="output"))
        model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                       metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        return model_

    # FFNN
    if architecture == 'FFNN':
        model_ = tf.keras.Sequential()
        model_.add(tf.keras.layers.Input(shape=(input_layer_size,)))
        for _ in range(config.hidden_layers):
            model_.add(tf.keras.layers.Dense(hidden_layer_size, activation='relu', use_bias=True))
        model_.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy", SparseTopKCategoricalAccuracy(
            k=3, name="k3_accuracy")])

    # CNN
    if architecture == 'CNN':
        config.FEATURE_ENGINEERING = False
        config.PAD_INPUT = True
        model_ = tf.keras.Sequential()
        model_.add(tf.keras.layers.Conv1D(
            filters=config.filters, kernel_size=config.kernel_size, input_shape=(args.max_train_len, 1), activation='relu'))
        for _ in range(config.layers - 1):
            model_.add(tf.keras.layers.Conv1D(filters=config.filters, kernel_size=config.kernel_size, activation='relu'))
        # model_.add(tf.keras.layers.Dropout(0.2))
        model_.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model_.add(tf.keras.layers.Flatten())
        model_.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy", SparseTopKCategoricalAccuracy(
            k=3, name="k3_accuracy")])

    # LSTM
    if architecture == 'LSTM':
        config.FEATURE_ENGINEERING = False
        config.PAD_INPUT = True
        model_ = tf.keras.Sequential()
        model_.add(tf.keras.layers.Embedding(56, 64, input_length=args.max_train_len))
        # model_.add(tf.keras.layers.Dropout(0.2))
        model_.add(tf.keras.layers.LSTM(config.lstm_units))
        # model_.add(tf.keras.layers.Dropout(0.2))
        model_.add(tf.keras.layers.Flatten())
        model_.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy", SparseTopKCategoricalAccuracy(
            k=3, name="k3_accuracy")])

    # Decision Tree
    if architecture == 'DT':
        model_ = DecisionTreeClassifier(criterion=config.criterion, ccp_alpha=config.ccp_alpha)

    # Naive Bayes
    if architecture == 'NB':
        model_ = MultinomialNB(alpha=config.alpha, fit_prior=config.fit_prior)

    # Random Forest
    if architecture == 'RF':
        model_ = RandomForestClassifier(n_estimators=config.n_estimators, criterion=config.criterion, bootstrap=config.bootstrap, n_jobs=30,
                                        max_features=config.max_features, max_depth=30, min_samples_split=config.min_samples_split,
                                        min_samples_leaf=config.min_samples_leaf)

    # Extra Trees
    if architecture == 'ET':
        model_ = ExtraTreesClassifier(n_estimators=config.n_estimators, criterion=config.criterion, bootstrap=config.bootstrap, n_jobs=30,
                                      max_features=config.max_features, max_depth=30, min_samples_split=config.min_samples_split,
                                      min_samples_leaf=config.min_samples_leaf)

    # Transformer
    if architecture == "Transformer":
        config.FEATURE_ENGINEERING = False
        config.PAD_INPUT = True
        vocab_size = config.vocab_size
        maxlen = args.max_train_len
        embed_dim = config.embed_dim  # Embedding size for each token
        num_heads = config.num_heads  # Number of attention heads
        ff_dim = config.ff_dim  # Hidden layer size in feed forward network inside transformer

        inputs = tf.keras.layers.Input(shape=(maxlen,))
        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(output_layer_size, activation="softmax")(x)

        model_ = tf.keras.Model(inputs=inputs, outputs=outputs)
        model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy", SparseTopKCategoricalAccuracy(
            k=3, name="k3_accuracy")])

    # FFNN, NB
    if architecture == "[FFNN,NB]":
        model_ffnn = tf.keras.Sequential()
        model_ffnn.add(tf.keras.layers.Input(shape=(input_layer_size,)))
        for _ in range(config.hidden_layers):
            model_ffnn.add(tf.keras.layers.Dense(hidden_layer_size, activation='relu', use_bias=True))
        model_ffnn.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model_ffnn.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                           metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        model_nb = MultinomialNB(alpha=config.alpha, fit_prior=config.fit_prior)
        return [model_ffnn, model_nb]
    return model_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CANN Ciphertype Detection Neuronal Network Training Script', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size for training.')
    parser.add_argument('--train_dataset_size', default=16000, type=int,
                        help='Dataset size per fit. This argument should be dividable \n'
                             'by the amount of --ciphers.')
    parser.add_argument('--dataset_workers', default=1, type=int,
                        help='The number of parallel workers for reading the \ninput files.')
    parser.add_argument('--epochs', default=1, type=int,
                        help='Defines how many times the same data is used to fit the model.')
    parser.add_argument('--input_directory', default='../data/gutenberg_en', type=str,
                        help='Input directory of the plaintexts.')
    parser.add_argument('--download_dataset', default=True, type=str2bool,
                        help='Download the dataset automatically.')
    parser.add_argument('--save_directory', default='../data/models/',
                        help='Directory for saving generated models. \n'
                             'When interrupting, the current model is \n'
                             'saved as interrupted_...')
    parser.add_argument('--model_name', default='m.h5', type=str,
                        help='Name of the output model file. The file must \nhave the .h5 extension.')
    parser.add_argument('--ciphers', default='aca', type=str,
                        help='A comma seperated list of the ciphers to be created.\n'
                             'Be careful to not use spaces or use \' to define the string.\n'
                             'Possible values are:\n'
                             '- mtc3 (contains the ciphers Monoalphabetic Substitution, Vigenere,\n'
                             '        Columnar Transposition, Plaifair and Hill)\n'
                             '- aca (contains all currently implemented ciphers from \n'
                             '       https://www.cryptogram.org/resource-area/cipher-types/)\n'
                             '- all aca ciphers in lower case'
                             '- simple_substitution\n'
                             '- vigenere\n'
                             '- columnar_transposition\n'
                             '- playfair\n'
                             '- hill\n')
    parser.add_argument('--keep_unknown_symbols', default=False, type=str2bool,
                        help='Keep unknown symbols in the plaintexts. Known \n'
                             'symbols are defined in the alphabet of the cipher.')
    parser.add_argument('--max_iter', default=1000000, type=int,
                        help='the maximal number of iterations before stopping training.')
    parser.add_argument('--min_train_len', default=50, type=int,
                        help='The minimum length of a plaintext to be encrypted in training. \n'
                             'If this argument is set to -1 no lower limit is used.')
    parser.add_argument('--min_test_len', default=50, type=int,
                        help='The minimum length of a plaintext to be encrypted in testing. \n'
                             'If this argument is set to -1 no lower limit is used.')
    parser.add_argument('--max_train_len', default=-1, type=int,
                        help='The maximum length of a plaintext to be encrypted in training. \n'
                             'If this argument is set to -1 no upper limit is used.')
    parser.add_argument('--max_test_len', default=-1, type=int,
                        help='The maximum length of a plaintext to be encrypted in testing. \n'
                             'If this argument is set to -1 no upper limit is used.')
    parser.add_argument('--architecture', default='FFNN', type=str, choices=['FFNN', 'CNN', 'LSTM', 'DT', 'NB', 'RF', 'ET', 'Transformer',
                        '[FFNN,NB]'],
                        help='The architecture to be used for training. \n'
                             'Possible values are:\n'
                             '- FFNN\n'
                             '- CNN\n'
                             '- LSTM\n'
                             '- DT\n'
                             '- NB\n'
                             '- RF\n'
                             '- ET\n'
                             '- Transformer\n'
                             '- [FFNN,NB]')
    parser.add_argument('--extend_model', default=None, type=str,
                        help='Load a trained model from a file and use it as basis for the new training.')

    args = parser.parse_args()
    for arg in vars(args):
        print("{:23s}= {:s}".format(arg, str(getattr(args, arg))))
    m = os.path.splitext(args.model_name)
    if len(os.path.splitext(args.model_name)) != 2 or os.path.splitext(args.model_name)[1] != '.h5':
        print('ERROR: The model name must have the ".h5" extension!', file=sys.stderr)
        sys.exit(1)
    args.input_directory = os.path.abspath(args.input_directory)
    args.ciphers = args.ciphers.lower()
    architecture = args.architecture
    cipher_types = args.ciphers.split(',')
    extend_model = args.extend_model
    if extend_model is not None:
        if architecture not in ('FFNN', 'CNN', 'LSTM'):
            print('ERROR: Models with the architecture %s can not be extended!' % architecture, file=sys.stderr)
            sys.exit(1)
        if len(os.path.splitext(extend_model)) != 2 or os.path.splitext(extend_model)[1] != '.h5':
            print('ERROR: The extended model name must have the ".h5" extension!', file=sys.stderr)
            sys.exit(1)

    if config.MTC3 in cipher_types:
        del cipher_types[cipher_types.index(config.MTC3)]
        cipher_types.append(config.CIPHER_TYPES[0])
        cipher_types.append(config.CIPHER_TYPES[1])
        cipher_types.append(config.CIPHER_TYPES[2])
        cipher_types.append(config.CIPHER_TYPES[3])
        cipher_types.append(config.CIPHER_TYPES[4])
    elif config.ACA in cipher_types:
        del cipher_types[cipher_types.index(config.ACA)]
        cipher_types.append(config.CIPHER_TYPES[0])
        cipher_types.append(config.CIPHER_TYPES[1])
        cipher_types.append(config.CIPHER_TYPES[2])
        cipher_types.append(config.CIPHER_TYPES[3])
        cipher_types.append(config.CIPHER_TYPES[4])
        cipher_types.append(config.CIPHER_TYPES[5])
        cipher_types.append(config.CIPHER_TYPES[6])
        cipher_types.append(config.CIPHER_TYPES[7])
        cipher_types.append(config.CIPHER_TYPES[8])
        cipher_types.append(config.CIPHER_TYPES[9])
        cipher_types.append(config.CIPHER_TYPES[10])
        cipher_types.append(config.CIPHER_TYPES[11])
        cipher_types.append(config.CIPHER_TYPES[12])
        cipher_types.append(config.CIPHER_TYPES[13])
        cipher_types.append(config.CIPHER_TYPES[14])
        cipher_types.append(config.CIPHER_TYPES[15])
        cipher_types.append(config.CIPHER_TYPES[16])
        cipher_types.append(config.CIPHER_TYPES[17])
        cipher_types.append(config.CIPHER_TYPES[18])
        cipher_types.append(config.CIPHER_TYPES[19])
        cipher_types.append(config.CIPHER_TYPES[20])
        cipher_types.append(config.CIPHER_TYPES[21])
        cipher_types.append(config.CIPHER_TYPES[22])
        cipher_types.append(config.CIPHER_TYPES[23])
        cipher_types.append(config.CIPHER_TYPES[24])
        cipher_types.append(config.CIPHER_TYPES[25])
        cipher_types.append(config.CIPHER_TYPES[26])
        cipher_types.append(config.CIPHER_TYPES[27])
        cipher_types.append(config.CIPHER_TYPES[28])
        cipher_types.append(config.CIPHER_TYPES[29])
        cipher_types.append(config.CIPHER_TYPES[30])
        cipher_types.append(config.CIPHER_TYPES[31])
        cipher_types.append(config.CIPHER_TYPES[32])
        cipher_types.append(config.CIPHER_TYPES[33])
        cipher_types.append(config.CIPHER_TYPES[34])
        cipher_types.append(config.CIPHER_TYPES[35])
        cipher_types.append(config.CIPHER_TYPES[36])
        cipher_types.append(config.CIPHER_TYPES[37])
        cipher_types.append(config.CIPHER_TYPES[38])
        cipher_types.append(config.CIPHER_TYPES[39])
        cipher_types.append(config.CIPHER_TYPES[40])
        cipher_types.append(config.CIPHER_TYPES[41])
        cipher_types.append(config.CIPHER_TYPES[42])
        cipher_types.append(config.CIPHER_TYPES[43])
        cipher_types.append(config.CIPHER_TYPES[44])
        cipher_types.append(config.CIPHER_TYPES[45])
        cipher_types.append(config.CIPHER_TYPES[46])
        cipher_types.append(config.CIPHER_TYPES[47])
        cipher_types.append(config.CIPHER_TYPES[48])
        cipher_types.append(config.CIPHER_TYPES[49])
        cipher_types.append(config.CIPHER_TYPES[50])
        cipher_types.append(config.CIPHER_TYPES[51])
        cipher_types.append(config.CIPHER_TYPES[52])
        cipher_types.append(config.CIPHER_TYPES[53])
        cipher_types.append(config.CIPHER_TYPES[54])
        cipher_types.append(config.CIPHER_TYPES[55])
    config.CIPHER_TYPES = cipher_types
    if args.train_dataset_size * args.dataset_workers > args.max_iter:
        print("ERROR: --train_dataset_size * --dataset_workers must not be bigger than --max_iter. "
              "In this case it was %d > %d" % (args.train_dataset_size * args.dataset_workers, args.max_iter), file=sys.stderr)
        sys.exit(1)

    if args.download_dataset and not os.path.exists(args.input_directory) and args.input_directory == os.path.abspath(
            '../data/gutenberg_en'):
        print("Downloading Datsets...")
        tfds.download.add_checksums_dir('../data/checksums/')
        download_manager = tfds.download.download_manager.DownloadManager(download_dir='../data/', extract_dir=args.input_directory)
        download_manager.download_and_extract('https://drive.google.com/uc?id=1bF5sSVjxTxa3DB-P5wxn87nxWndRhK_V&export=download')
        path = os.path.join(args.input_directory, 'ZIP.ucid_1bF5sSVjxTx-P5wxn87nxWn_V_export_downloadR9Cwhunev5CvJ-ic__'
                                                  'HawxhTtGOlSdcCrro4fxfEI8A', os.path.basename(args.input_directory))
        dir_name = os.listdir(path)
        for name in dir_name:
            p = Path(os.path.join(path, name))
            parent_dir = p.parents[2]
            p.rename(parent_dir / p.name)
        os.rmdir(path)
        os.rmdir(os.path.dirname(path))
        print("Datasets Downloaded.")

    print("Loading Datasets...")
    plaintext_files = []
    dir_name = os.listdir(args.input_directory)
    for name in dir_name:
        path = os.path.join(args.input_directory, name)
        if os.path.isfile(path):
            plaintext_files.append(path)
    train, test = train_test_split(plaintext_files, test_size=0.05, random_state=42, shuffle=True)

    train_ds = TextLine2CipherStatisticsDataset(train, cipher_types, args.train_dataset_size, args.min_train_len, args.max_train_len,
                                                args.keep_unknown_symbols, args.dataset_workers)
    test_ds = TextLine2CipherStatisticsDataset(test, cipher_types, args.train_dataset_size, args.min_test_len, args.max_test_len,
                                               args.keep_unknown_symbols, args.dataset_workers)
    if args.train_dataset_size % train_ds.key_lengths_count != 0:
        print("WARNING: the --train_dataset_size parameter must be dividable by the amount of --ciphers  and the length configured "
              "KEY_LENGTHS in config.py. The current key_lengths_count is %d" % train_ds.key_lengths_count, file=sys.stderr)
    print("Datasets loaded.\n")

    print('Creating model...')

    gpu_count = len(tf.config.list_physical_devices('GPU')) + len(tf.config.list_physical_devices('XLA_GPU'))
    if gpu_count > 1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            if extend_model is not None:
                extend_model = tf.keras.models.load_model(extend_model, compile=False)
            model = create_model()
        if architecture in ("FFNN", "CNN", "LSTM", "Transformer") and extend_model is None:
            model.summary()
    else:
        if extend_model is not None:
            extend_model = tf.keras.models.load_model(extend_model, compile=False)
        model = create_model()
        if architecture in ("FFNN", "CNN", "LSTM", "Transformer") and extend_model is None:
        model.summary()

    print('Model created.\n')

    print('Training model...')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../data/logs', update_freq='epoch')
    early_stopping_callback = MiniBatchEarlyStopping(
        min_delta=1e-5, patience=250, monitor='accuracy', mode='max', restore_best_weights=True)
    # time_based_decay_lrate_callback = TimeBasedDecayLearningRateScheduler(args.train_dataset_size)
    custom_step_decay_lrate_callback = CustomStepDecayLearningRateScheduler(early_stopping_callback)
    start_time = time.time()
    cntr = 0
    train_iter = 0
    train_epoch = 0
    val_data = None
    val_labels = None
    run = None
    run1 = None
    processes = []
    classes = list(range(len(config.CIPHER_TYPES)))
    new_run = [[], []]
    while train_ds.iteration < args.max_iter:
        if run1 is None:
            train_epoch = 0
            processes, run1 = train_ds.__next__()
        if run is None:
            for process in processes:
                process.join()
            run = run1
            train_ds.iteration += train_ds.batch_size * train_ds.dataset_workers
            if train_ds.iteration < args.max_iter:
                train_epoch = train_ds.epoch
                processes, run1 = train_ds.__next__()
        # use this only with decision trees
        if architecture in ("DT", "RF", "ET"):
            for batch, labels in run:
                new_run[0].extend(batch.numpy().tolist())
                new_run[1].extend(labels.numpy().tolist())
            if train_ds.iteration < args.max_iter:
                run = None
                print("Loaded %d ciphertexts." % train_ds.iteration)
                continue
            print("Loaded %d ciphertexts." % train_ds.iteration)
            for process in processes:
                if process.is_alive():
                    process.terminate()
            new_run = [(tf.convert_to_tensor(new_run[0]), tf.convert_to_tensor(new_run[1]))]
            run = new_run
        for batch, labels in run:
            cntr += 1
            train_iter = args.train_dataset_size * cntr
            if cntr == 1:
                batch, val_data, labels, val_labels = train_test_split(batch.numpy(), labels.numpy(), test_size=0.1)
                batch = tf.convert_to_tensor(batch)
                val_data = tf.convert_to_tensor(val_data)
                labels = tf.convert_to_tensor(labels)
                val_labels = tf.convert_to_tensor(val_labels)
            train_iter -= args.train_dataset_size * 0.1

            # Decision Tree training
            if architecture in ("DT", "RF", "ET"):
                train_iter = len(labels) * 0.9
                print("Start training the decision tree.")
                history = model.fit(batch, labels)
                if architecture == "DT":
                    plt.gcf().set_size_inches(25, 25 / math.sqrt(2))
                    print("Plotting tree.")
                    plot_tree(model, max_depth=3, fontsize=6, filled=True)
                    plt.savefig(args.model_name.split('.')[0] + '_decision_tree.svg', dpi=200, bbox_inches='tight', pad_inches=0.1)

            # Naive Bayes training
            elif architecture == "NB":
                history = model.partial_fit(batch, labels, classes=classes)

            # FFNN, NB
            elif architecture == "[FFNN,NB]":
                history = model[0].fit(batch, labels, batch_size=args.batch_size, validation_data=(val_data, val_labels), epochs=args.epochs,
                                    callbacks=[early_stopping_callback, tensorboard_callback, custom_step_decay_lrate_callback])
                # time_based_decay_lrate_callback.iteration = train_iter
                history = model[1].partial_fit(batch, labels, classes=classes)

            else:
                history = model.fit(batch, labels, batch_size=args.batch_size, validation_data=(val_data, val_labels), epochs=args.epochs,
                                    callbacks=[early_stopping_callback, tensorboard_callback, custom_step_decay_lrate_callback])
                # time_based_decay_lrate_callback.iteration = train_iter

            # print for Decision Tree, Naive Bayes and Random Forests
            if architecture in ("DT", "NB", "RF", "ET"):
                val_score = model.score(val_data, val_labels)
                train_score = model.score(batch, labels)
                print("train accuracy: %f, validation accuracy: %f" % (train_score, val_score))

            if architecture == "[FFNN,NB]":
                val_score = model[1].score(val_data, val_labels)
                train_score = model[1].score(batch, labels)
                print("train accuracy: %f, validation accuracy: %f" % (train_score, val_score))

            if train_epoch > 0:
                train_epoch = train_iter // ((train_ds.iteration + train_ds.batch_size * train_ds.dataset_workers) // train_ds.epoch)
            print("Epoch: %d, Iteration: %d" % (train_epoch, train_iter))
            if train_iter >= args.max_iter or early_stopping_callback.stop_training:
                break
        if train_ds.iteration >= args.max_iter or early_stopping_callback.stop_training:
            break
        run = None
    for process in processes:
        if process.is_alive():
            process.terminate()

    elapsed_training_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)
    print('Finished training in %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.\n' % (
        elapsed_training_time.days, elapsed_training_time.seconds // 3600, (elapsed_training_time.seconds // 60) % 60,
        elapsed_training_time.seconds % 60, train_iter, train_epoch))

    print('Saving model...')
    if args.model_name == 'm.h5':
        i = 1
        while os.path.exists(os.path.join(args.save_directory, args.model_name.split('.')[0] + str(i) + '.h5')):
            i += 1
        model_name = args.model_name.split('.')[0] + str(i) + '.h5'
    else:
        model_name = args.model_name
    model_path = os.path.join(args.save_directory, model_name)
    if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
        model.save(model_path)
    elif architecture in ("DT", "NB", "RF", "ET"):
        with open(model_path, "wb") as f:
            # this gets very large
            pickle.dump(model, f)
    elif architecture == "[FFNN,NB]":
        model[0].save('../data/models/' + model_path.split('.')[0] + "_ffnn.h5")
        with open('../data/models/' + model_path.split('.')[0] + "_nb.h5", "wb") as f:
            # this gets very large
            pickle.dump(model[1], f)
    with open('../data/' + model_path.split('.')[0] + '_parameters.txt', 'w') as f:
        for arg in vars(args):
            f.write("{:23s}= {:s}\n".format(arg, str(getattr(args, arg))))
    if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
        shutil.move('../data/logs', '../data/' + model_name.split('.')[0] + '_tensorboard_logs')
    print('Model saved.\n')

    print('Predicting test data...\n')
    start_time = time.time()
    correct = [0]*len(config.CIPHER_TYPES)
    total = [0]*len(config.CIPHER_TYPES)
    correct_all = 0
    total_len_prediction = 0
    incorrect = []
    for i in range(len(config.CIPHER_TYPES)):
        incorrect += [[0]*len(config.CIPHER_TYPES)]

    prediction_dataset_factor = 10
    if early_stopping_callback.stop_training:
        while test_ds.dataset_workers * test_ds.batch_size > train_iter / prediction_dataset_factor and prediction_dataset_factor > 1:
            prediction_dataset_factor -= 1
        args.max_iter = int(train_iter / prediction_dataset_factor)
    else:
        while test_ds.dataset_workers * test_ds.batch_size > args.max_iter / prediction_dataset_factor:
            prediction_dataset_factor -= 1
        args.max_iter /= prediction_dataset_factor
    cntr = 0
    test_iter = 0
    test_epoch = 0
    run = None
    run1 = None
    processes = []
    while test_ds.iteration < args.max_iter:
        if run1 is None:
            processes, run1 = test_ds.__next__()
        if run is None:
            for process in processes:
                process.join()
            run = run1
            test_ds.iteration += test_ds.batch_size * test_ds.dataset_workers
            if test_ds.iteration < args.max_iter:
                processes, run1 = test_ds.__next__()
        for batch, labels in run:
            # Decision Tree, Naive Bayes prediction
            if architecture in ("DT", "NB", "RF", "ET"):
                prediction = model.predict_proba(batch)
            elif architecture == "[FFNN,NB]":
                prediction = model[0].predict(batch, batch_size=args.batch_size, verbose=1)
            else:
                prediction = model.predict(batch, batch_size=args.batch_size, verbose=1)
            for i in range(len(prediction)):
                if labels[i] == np.argmax(prediction[i]):
                    correct_all += 1
                    correct[labels[i]] += 1
                else:
                    incorrect[labels[i]][np.argmax(prediction[i])] += 1
                total[labels[i]] += 1
            total_len_prediction += len(prediction)
            cntr += 1
            test_iter = args.train_dataset_size * cntr
            test_epoch = test_ds.epoch
            if test_epoch > 0:
                test_epoch = test_iter // ((test_ds.iteration + test_ds.batch_size * test_ds.dataset_workers) // test_ds.epoch)
            print("Prediction Epoch: %d, Iteration: %d / %d" % (test_epoch, test_iter, args.max_iter))
            if test_iter >= args.max_iter:
                break
        if test_ds.iteration >= args.max_iter:
            break
        run = None
    for process in processes:
        if process.is_alive():
            process.terminate()

    elapsed_prediction_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)

    if total_len_prediction > args.train_dataset_size:
        total_len_prediction -= total_len_prediction % args.train_dataset_size
    print('\ntest data predicted: %d ciphertexts' % total_len_prediction)
    for i in range(0, len(total)):
        if total[i] == 0:
            continue
        print('%s correct: %d/%d = %f' % (config.CIPHER_TYPES[i], correct[i], total[i], correct[i] / total[i]))
    if total_len_prediction == 0:
        t = 'N/A'
    else:
        t = str(correct_all / total_len_prediction)
    print('Total: %s\n' % t)
    print('Finished training in %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.\n' % (
        elapsed_training_time.days, elapsed_training_time.seconds // 3600, (elapsed_training_time.seconds // 60) % 60,
        elapsed_training_time.seconds % 60, train_iter, train_epoch))
    print('Prediction time: %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.' % (
        elapsed_prediction_time.days, elapsed_prediction_time.seconds // 3600, (elapsed_prediction_time.seconds // 60) % 60,
        elapsed_prediction_time.seconds % 60, test_iter, test_epoch))

    print("Incorrect prediction counts: %s" % incorrect)
