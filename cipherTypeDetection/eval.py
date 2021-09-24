from pathlib import Path
import argparse
import sys
import os
import pickle
import ast
import functools
import numpy as np
from datetime import datetime
# This environ variable must be set before all tensorflow imports!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
sys.path.append("../")
from util.utils import map_text_into_numberspace
from util.utils import print_progress
import cipherTypeDetection.config as config
from cipherTypeDetection.textLine2CipherStatisticsDataset import TextLine2CipherStatisticsDataset, calculate_statistics, pad_sequences
from cipherTypeDetection.ensembleModel import EnsembleModel
from cipherTypeDetection.transformer import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding
from util.utils import get_model_input_length
from cipherImplementations.cipher import OUTPUT_ALPHABET, UNKNOWN_SYMBOL_NUMBER
tf.debugging.set_log_device_placement(enabled=False)
# always flush after print as some architectures like RF need very long time before printing anything.
print = functools.partial(print, flush=True)


architecture = None
model_path = None
model_list = None
architecture_list = None
strategy = None
cipher_types = None


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def benchmark(args_, model_):
    args_.plaintext_folder = os.path.abspath(args_.plaintext_folder)
    if args_.dataset_size * args_.dataset_workers > args_.max_iter:
        print("ERROR: --dataset_size * --dataset_workers must not be bigger than --max_iter. In this case it was %d > %d" % (
            args_.dataset_size * args_.dataset_workers, args_.max_iter), file=sys.stderr)
        sys.exit(1)
    if args_.download_dataset and not os.path.exists(args_.plaintext_folder) and args_.plaintext_folder == os.path.abspath(
            '../data/gutenberg_en'):
        print("Downloading Datsets...")
        tfds.download.add_checksums_dir('../data/checksums/')
        download_manager = tfds.download.download_manager.DownloadManager(download_dir='../data/', extract_dir=args_.plaintext_folder)
        download_manager.download_and_extract(
            'https://drive.google.com/uc?id=1bF5sSVjxTxa3DB-P5wxn87nxWndRhK_V&export=download')
        path = os.path.join(args_.plaintext_folder, 'ZIP.ucid_1bF5sSVjxTx-P5wxn87nxWn_V_export_downloadR9Cwhunev5CvJ-ic__'
                                                    'HawxhTtGOlSdcCrro4fxfEI8A', os.path.basename(args_.plaintext_folder))
        dir_nam = os.listdir(path)
        for name in dir_nam:
            p = Path(os.path.join(path, name))
            parent_dir = p.parents[2]
            p.rename(parent_dir / p.name)
        os.rmdir(path)
        os.rmdir(os.path.dirname(path))
        print("Datasets Downloaded.")

    print("Loading Datasets...")
    plaintext_files = []
    dir_nam = os.listdir(args_.plaintext_folder)
    for name in dir_nam:
        path = os.path.join(args_.plaintext_folder, name)
        if os.path.isfile(path):
            plaintext_files.append(path)
    dataset = TextLine2CipherStatisticsDataset(plaintext_files, cipher_types, args_.dataset_size, args_.min_text_len, args_.max_text_len,
                                               args_.keep_unknown_symbols, args_.dataset_workers, generate_test_data=True)
    if args_.dataset_size % dataset.key_lengths_count != 0:
        print("WARNING: the --dataset_size parameter must be dividable by the amount of --ciphers  and the length configured KEY_LENGTHS in"
              " config.py. The current key_lengths_count is %d" % dataset.key_lengths_count, file=sys.stderr)
    print("Datasets loaded.\n")

    print('Evaluating model...')
    import time
    start_time = time.time()
    cntr = 0
    iteration = 0
    epoch = 0
    results = []
    run = None
    run1 = None
    ciphertexts = None
    ciphertexts1 = None
    processes = []
    while dataset.iteration < args_.max_iter:
        if run1 is None:
            epoch = 0
            processes, run1, ciphertexts1 = dataset.__next__()
        if run is None:
            for process in processes:
                process.join()
            run = run1
            ciphertexts = ciphertexts1
            dataset.iteration += dataset.batch_size * dataset.dataset_workers
            if dataset.iteration < args_.max_iter:
                epoch = dataset.epoch
                processes, run1, ciphertexts1 = dataset.__next__()
        for j in range(len(run)):
            batch, labels = run[j]
            batch = tf.convert_to_tensor(batch)
            labels = tf.convert_to_tensor(labels)
            batch_ciphertexts = tf.convert_to_tensor(ciphertexts[j])
            if architecture == "FFNN":
                results.append(model_.evaluate(batch, labels, batch_size=args_.batch_size, verbose=1))
            if architecture in ("CNN", "LSTM", "Transformer"):
                results.append(model_.evaluate(batch_ciphertexts, labels, batch_size=args_.batch_size, verbose=1))
            elif architecture in ("DT", "NB", "RF", "ET"):
                results.append(model_.score(batch, labels))
                print("accuracy: %f" % (results[-1]))
            elif architecture == "Ensemble":
                results.append(model_.evaluate(batch, batch_ciphertexts, labels, args_.batch_size, verbose=1))
            cntr += 1
            iteration = args_.dataset_size * cntr
            epoch = dataset.epoch
            if epoch > 0:
                epoch = iteration // (dataset.iteration // dataset.epoch)
            print("Epoch: %d, Iteration: %d" % (epoch, iteration))
            if iteration >= args_.max_iter:
                break
        run = None
        if dataset.iteration >= args_.max_iter:
            break
    elapsed_evaluation_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)
    print('Finished evaluation in %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.\n' % (
        elapsed_evaluation_time.days, elapsed_evaluation_time.seconds // 3600, (elapsed_evaluation_time.seconds // 60) % 60,
        elapsed_evaluation_time.seconds % 60, iteration, epoch))

    if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
        avg_loss = 0
        avg_acc = 0
        avg_k3_acc = 0
        for loss, acc_pred, k3_acc in results:
            avg_loss += loss
            avg_acc += acc_pred
            avg_k3_acc += k3_acc
        avg_loss = avg_loss / len(results)
        avg_acc = avg_acc / len(results)
        avg_k3_acc = avg_k3_acc / len(results)
        print("Average evaluation results: loss: %f, accuracy: %f, k3_accuracy: %f\n" % (avg_loss, avg_acc, avg_k3_acc))
    elif architecture in ("DT", "NB", "RF", "ET", "Ensemble"):
        avg_test_acc = 0
        for acc in results:
            avg_test_acc += acc
        avg_test_acc = avg_test_acc / len(results)
        print("Average evaluation results from %d iterations: avg_test_acc=%f" % (iteration, avg_test_acc))


def evaluate(args_, model_):
    results_list = []
    dir_name = os.listdir(args_.data_folder)
    dir_name.sort()
    cntr = 0
    iterations = 0
    for name in dir_name:
        if iterations > args_.max_iter:
            break
        path = os.path.join(args_.data_folder, name)
        if os.path.isfile(path):
            if iterations > args_.max_iter:
                break
            batch = []
            batch_ciphertexts = []
            labels = []
            results = []
            dataset_cnt = 0
            input_length = get_model_input_length(model_, args_.architecture)
            with open(path, "rb") as fd:
                lines = fd.readlines()
            for line in lines:
                # remove newline
                line = line.strip(b'\n').decode()
                if line == '':
                    continue
                split_line = line.split(' ')
                labels.append(int(split_line[0]))
                statistics = [float(f) for f in split_line[1].split(',')]
                batch.append(statistics)
                ciphertext = [int(j) for j in split_line[2].split(',')]
                if input_length is not None:
                    if len(ciphertext) < input_length:
                        ciphertext = pad_sequences([ciphertext], maxlen=input_length)[0]
                    # if the length its too high, we need to strip it..
                    elif len(ciphertext) > input_length:
                        ciphertext = ciphertext[:input_length]
                batch_ciphertexts.append(ciphertext)
                iterations += 1
                if iterations == args_.max_iter:
                    break
                if len(labels) == args_.dataset_size:
                    if architecture == "FFNN":
                        results.append(model_.evaluate(tf.convert_to_tensor(batch), tf.convert_to_tensor(labels), args_.batch_size, verbose=0))
                    elif architecture in ("CNN", "LSTM", "Transformer"):
                        results.append(model_.evaluate(tf.convert_to_tensor(batch_ciphertexts), tf.convert_to_tensor(labels),
                                       args_.batch_size, verbose=0))
                    elif architecture == "Ensemble":
                        results.append(model_.evaluate(tf.convert_to_tensor(batch), tf.convert_to_tensor(batch_ciphertexts), tf.convert_to_tensor(labels),
                                       args_.batch_size, verbose=0))
                    elif architecture in ("DT", "NB", "RF", "ET"):
                        results.append(model.score(batch, tf.convert_to_tensor(labels)))
                    batch = []
                    batch_ciphertexts = []
                    labels = []
                    dataset_cnt += 1
            if len(labels) > 0:
                if architecture == "FFNN":
                    results.append(model_.evaluate(tf.convert_to_tensor(batch), tf.convert_to_tensor(labels), args_.batch_size, verbose=0))
                elif architecture in ("CNN", "LSTM", "Transformer"):
                    results.append(
                        model_.evaluate(tf.convert_to_tensor(batch_ciphertexts), tf.convert_to_tensor(labels), args_.batch_size, verbose=0))
                elif architecture == "Ensemble":
                    results.append(
                        model_.evaluate(tf.convert_to_tensor(batch), tf.convert_to_tensor(batch_ciphertexts), tf.convert_to_tensor(labels),
                                        args_.batch_size, verbose=0))
                elif architecture in ("DT", "NB", "RF", "ET"):
                    results.append(model.score(batch, tf.convert_to_tensor(labels)))
            if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
                avg_loss = 0
                avg_acc = 0
                avg_k3_acc = 0
                for loss, acc_pred, k3_acc in results:
                    avg_loss += loss
                    avg_acc += acc_pred
                    avg_k3_acc += k3_acc
                result = [avg_loss / len(results), avg_acc / len(results), avg_k3_acc / len(results)]
            elif architecture in ("DT", "NB", "RF", "ET", "Ensemble"):
                avg_test_acc = 0
                for acc in results:
                    avg_test_acc += acc
                result = avg_test_acc / len(results)
            results_list.append(result)
            cntr += 1
            if args_.evaluation_mode == 'per_file':
                if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
                    print("%s (%d lines) test_loss: %f, test_accuracy: %f, test_k3_accuracy: %f (progress: %d%%)" % (
                        os.path.basename(path), len(batch) + dataset_cnt * args_.dataset_size, result[0], result[1], result[2], max(
                            int(cntr / len(dir_name) * 100), int(iterations / args_.max_iter) * 100)))
                elif architecture in ("DT", "NB", "RF", "ET", "Ensemble"):
                    print("%s (%d lines) test_accuracy: %f (progress: %d%%)" % (
                        os.path.basename(path), len(batch) + dataset_cnt * args_.dataset_size, result,
                        max(int(cntr / len(dir_name) * 100), int(iterations / args_.max_iter) * 100)))
            else:
                print_progress("Evaluating files: ", cntr, len(dir_name), factor=5)
            if iterations == args_.max_iter:
                break

    if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
        avg_test_loss = 0
        avg_test_acc = 0
        avg_test_acc_k3 = 0
        for loss, acc, acc_k3 in results_list:
            avg_test_loss += loss
            avg_test_acc += acc
            avg_test_acc_k3 += acc_k3
        avg_test_loss = avg_test_loss / len(results_list)
        avg_test_acc = avg_test_acc / len(results_list)
        avg_test_acc_k3 = avg_test_acc_k3 / len(results_list)
        print("\n\nAverage evaluation results from %d iterations: avg_test_loss=%f, avg_test_acc=%f, avg_test_acc_k3=%f" % (
            iterations, avg_test_loss, avg_test_acc, avg_test_acc_k3))
    elif architecture in ("DT", "NB", "RF", "ET", "Ensemble"):
        avg_test_acc = 0
        for acc in results_list:
            avg_test_acc += acc
        avg_test_acc = avg_test_acc / len(results_list)
        print("\n\nAverage evaluation results from %d iterations: avg_test_acc=%f" % (iterations, avg_test_acc))


def predict_single_line(args_, model_):
    config.CIPHER_TYPES = args_.ciphers
    cipher_id_result = ''
    ciphertexts = []
    result = []
    if args_.ciphertext is not None:
        ciphertexts.append(args_.ciphertext.encode())
    else:
        ciphertexts = open(args_.file, 'rb')

    print()
    for line in ciphertexts:
        # remove newline
        line = line.strip(b'\n')
        if line == b'':
            continue
        # evaluate aca features file
        # label = line.split(b' ')[0]
        # statistics = ast.literal_eval(line.split(b' ')[1].decode())
        # ciphertext = ast.literal_eval(line.split(b' ')[2].decode())
        # print(config.CIPHER_TYPES[int(label.decode())], "length: %d" % len(ciphertext))

        print(line)
        ciphertext = map_text_into_numberspace(line, OUTPUT_ALPHABET, UNKNOWN_SYMBOL_NUMBER)
        try:
            statistics = calculate_statistics(ciphertext)
        except ZeroDivisionError:
            print("\n")
            continue
        results = None
        if architecture == "FFNN":
            result = model_.predict(tf.convert_to_tensor([statistics]), args_.batch_size, verbose=0)
        elif architecture in ("CNN", "LSTM", "Transformer"):
            input_length = get_model_input_length(model_, architecture)
            if len(ciphertext) < input_length:
                ciphertext = pad_sequences([list(ciphertext)], maxlen=input_length)[0]
            split_ciphertext = [ciphertext[input_length*j:input_length*(j+1)] for j in range(len(ciphertext) // input_length)]
            results = []
            if architecture in ("LSTM", "Transformer"):
                for ct in split_ciphertext:
                    results.append(model_.predict(tf.convert_to_tensor([ct]), args_.batch_size, verbose=0))
            elif architecture == "CNN":
                for ct in split_ciphertext:
                    results.append(
                        model_.predict(tf.reshape(tf.convert_to_tensor([ct]), (1, input_length, 1)), args_.batch_size, verbose=0))
            result = results[0]
            for res in results[1:]:
                result = np.add(result, res)
            result = np.divide(result, len(results))
        elif architecture in ("DT", "NB", "RF", "ET"):
            result = model_.predict_proba(tf.convert_to_tensor([statistics]))
        elif architecture == "Ensemble":
            result = model_.predict(tf.convert_to_tensor([statistics]), [ciphertext], args_.batch_size, verbose=0)

        if isinstance(result, list):
            result_list = result[0]
        else:
            result_list = result[0].tolist()
        if results is not None and architecture not in ('Ensemble', 'LSTM', 'Transformer', 'CNN'):
            for j in range(len(result_list)):
                result_list[j] /= len(results)
        if args_.verbose:
            for cipher in args_.ciphers:
                print("{:23s} {:f}%".format(cipher, result_list[config.CIPHER_TYPES.index(cipher)]*100))
            max_val = max(result_list)
            cipher = config.CIPHER_TYPES[result_list.index(max_val)]
        else:
            max_val = max(result_list)
            cipher = config.CIPHER_TYPES[result_list.index(max_val)]
            print("{:s} {:f}%".format(cipher, max_val * 100))
        print()
        cipher_id_result += cipher[0].upper()

    if args_.file is not None:
        ciphertexts.close()

    # return a list of probabilities (does only return the last one in case a file is used)
    res_dict = {}
    if result:
        for j, val in enumerate(result[0]):
            res_dict[args_.ciphers[j]] = val * 100
    return res_dict


def load_model():
    global architecture
    if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
        if architecture == 'Transformer':
            if not hasattr(config, "maxlen"):
                raise ValueError("maxlen must be defined in the config when loading a Transformer model!")
            model_ = tf.keras.models.load_model(args.model, custom_objects={
                'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'MultiHeadSelfAttention': MultiHeadSelfAttention,
                'TransformerBlock': TransformerBlock})
        else:
            model_ = tf.keras.models.load_model(args.model)
        if architecture in ("CNN", "LSTM", "Transformer"):
            config.FEATURE_ENGINEERING = False
            config.PAD_INPUT = True
        else:
            config.FEATURE_ENGINEERING = True
            config.PAD_INPUT = False
        optimizer = Adam(learning_rate=config.learning_rate, beta_1=config.beta_1, beta_2=config.beta_2, epsilon=config.epsilon,
                         amsgrad=config.amsgrad)
        model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                       metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        return model_
    if architecture in ("DT", "NB", "RF", "ET"):
        config.FEATURE_ENGINEERING = True
        config.PAD_INPUT = False
        global model_path
        with open(model_path, "rb") as f:
            return pickle.load(f)
    if architecture == 'Ensemble':
        global model_list
        global architecture_list
        global strategy
        global cipher_types
        cipher_indices = []
        for cipher_type in cipher_types:
            cipher_indices.append(config.CIPHER_TYPES.index(cipher_type))
        return EnsembleModel(model_list, architecture_list, strategy, cipher_indices)
    else:
        raise ValueError("Unknown architecture: %s" % architecture)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CANN Ciphertype Detection Neuronal Network Evaluation Script', formatter_class=argparse.RawTextHelpFormatter)
    sp = parser.add_subparsers()
    bench_parser = sp.add_parser('benchmark',
                                 help='Use this argument to create ciphertexts on the fly, \nlike in training mode, and evaluate them with '
                                      'the model. \nThis option is optimized for large throughput to test the model.')
    eval_parser = sp.add_parser('evaluate', help='Use this argument to evaluate cipher types for single files or directories.')
    single_line_parser = sp.add_parser('single_line', help='Use this argument to predict a single line of ciphertext.')

    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size for training.')
    parser.add_argument('--max_iter', default=1000000000, type=int,
                        help='the maximal number of iterations before stopping evaluation.')
    parser.add_argument('--model', default='../data/models/m1.h5', type=str,
                        help='Name of the model file. The file must have the .h5 extension.')
    parser.add_argument('--architecture', default='FFNN', type=str, choices=[
        'FFNN', 'CNN', 'LSTM', 'DT', 'NB', 'RF', 'ET', 'Transformer', 'Ensemble'],
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
             '- Ensemble')
    parser.add_argument('--ciphers', '--ciphers', default='aca', type=str,
                        help='A comma seperated list of the ciphers to be created.\n'
                             'Be careful to not use spaces or use \' to define the string.\n'
                             'Possible values are:\n'
                             '- mtc3 (contains the ciphers Monoalphabetic Substitution, Vigenere,\n'
                             '        Columnar Transposition, Plaifair and Hill)\n'
                             '- aca (contains all currently implemented ciphers from \n'
                             '       https://www.cryptogram.org/resource-area/cipher-types/)\n'
                             '- simple_substitution\n'
                             '- vigenere\n'
                             '- columnar_transposition\n'
                             '- playfair\n'
                             '- hill\n')

    parser.add_argument('--models', nargs='+', default=None,
                        help='A list of models to be used in the ensemble model. The length of the list must be the same like the one in '
                             'the --architectures argument.')
    parser.add_argument('--architectures', nargs='+', default=None,
                        help='A list of the architectures to be used in the ensemble model. The length of the list must be the same like '
                             'the one in the --models argument.')
    parser.add_argument('--strategy', default='weighted', type=str, choices=['mean', 'weighted'],
                        help='The algorithm used for decisions.\n- Mean voting adds the probabilities from every class and returns the mean'
                             ' value of it. The highest value wins.\n- Weighted voting uses pre-calculated statistics, like for example '
                             'precision, to weight the output of a specific model for a specific class.')
    parser.add_argument('--dataset_size', default=16000, type=int,
                        help='Dataset size per evaluation. This argument should be dividable \nby the amount of --ciphers.')

    bench_parser.add_argument('--download_dataset', default=True, type=str2bool)
    bench_parser.add_argument('--dataset_workers', default=1, type=int)
    bench_parser.add_argument('--plaintext_folder', default='../data/gutenberg_en', type=str)
    bench_parser.add_argument('--keep_unknown_symbols', default=False, type=str2bool)
    bench_parser.add_argument('--min_text_len', default=50, type=int)
    bench_parser.add_argument('--max_text_len', default=-1, type=int)

    bench_group = parser.add_argument_group('benchmark')
    bench_group.add_argument('--download_dataset', help='Download the dataset automatically.')
    bench_group.add_argument('--dataset_workers', help='The number of parallel workers for reading the input files.')
    bench_group.add_argument('--plaintext_folder', help='Input folder of the plaintexts.')
    bench_group.add_argument('--keep_unknown_symbols', help='Keep unknown symbols in the plaintexts. Known \n'
                                                            'symbols are defined in the alphabet of the cipher.')
    bench_group.add_argument('--min_text_len', help='The minimum length of a plaintext to be encrypted in the evaluation process.\n'
                             'If this argument is set to -1 no lower limit is used.')
    bench_group.add_argument('--max_text_len', help='The maximum length of a plaintext to be encrypted in the evaluation process.\n'
                             'If this argument is set to -1 no upper limit is used.')

    eval_parser.add_argument('--evaluation_mode', nargs='?', choices=('summarized', 'per_file'), default='summarized', type=str)
    eval_parser.add_argument('--data_folder', default='../data/gutenberg_en', type=str)

    eval_group = parser.add_argument_group('evaluate')
    eval_group.add_argument('--evaluation_mode',
                            help='- To create an single evaluation result over all iterated data files use the \'summarized\' option.'
                                 '\n  This option is to be preferred over the benchmark option, if the tests should be reproducable.\n'
                                 '- To create an evaluation for every file use \'per_file\' option. This mode allows the \n'
                                 '  calculation of the \n  - average value of the prediction \n'
                                 '  - lower quartile - value at the position of 25 percent of the sorted predictions\n'
                                 '  - median - value at the position of 50 percent of the sorted predictions\n'
                                 '  - upper quartile - value at the position of 75 percent of the sorted predictions\n'
                                 '  With these statistics an expert can classify a ciphertext document to a specific cipher.')
    eval_group.add_argument('--data_folder', help='Input folder of the data files with labels and calculated features.')

    single_line_parser.add_argument('--verbose', default=True, type=str2bool)
    data = single_line_parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--ciphertext', default=None, type=str)
    data.add_argument('--file', default=None, type=str)

    single_line_group = parser.add_argument_group('single_line')
    single_line_group.add_argument('--ciphertext', help='A single line of ciphertext to be predicted by the model.')
    single_line_group.add_argument('--file', help='A file with lines of ciphertext to be predicted line by line by the model.')
    single_line_group.add_argument('--verbose', help='If true all predicted ciphers are printed. \n'
                                                     'If false only the most accurate prediction is printed.')

    args = parser.parse_args()
    for arg in vars(args):
        print("{:23s}= {:s}".format(arg, str(getattr(args, arg))))
    m = os.path.splitext(args.model)
    if len(os.path.splitext(args.model)) != 2 or os.path.splitext(args.model)[1] != '.h5':
        print('ERROR: The model name must have the ".h5" extension!', file=sys.stderr)
        sys.exit(1)

    architecture = args.architecture
    model_path = args.model
    args.ciphers = args.ciphers.lower()
    cipher_types = args.ciphers.split(',')
    if config.MTC3 in cipher_types:
        del cipher_types[cipher_types.index(config.MTC3)]
        cipher_types.append(config.CIPHER_TYPES[0])
        cipher_types.append(config.CIPHER_TYPES[1])
        cipher_types.append(config.CIPHER_TYPES[2])
        cipher_types.append(config.CIPHER_TYPES[3])
        cipher_types.append(config.CIPHER_TYPES[4])
    if config.ACA in cipher_types:
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
    args.ciphers = cipher_types
    if architecture == 'Ensemble':
        if not hasattr(args, 'models') or not hasattr(args, 'architectures'):
            raise ValueError("Please use the 'ensemble' subroutine if specifying the ensemble architecture.")
        if len(args.models) != len(args.architectures):
            raise ValueError("The length of --models must be the same like the length of --architectures.")
        models = []
        for i in range(len(args.models)):
            model = args.models[i]
            arch = args.architectures[i]
            if not os.path.exists(os.path.abspath(model)):
                raise ValueError("Model in %s does not exist." % os.path.abspath(model))
            if arch not in ('FFNN', 'CNN', 'LSTM', 'DT', 'NB', 'RF', 'ET', 'Transformer'):
                raise ValueError("Unallowed architecture %s" % arch)
            if arch in ('FFNN', 'CNN', 'LSTM', 'Transformer') and not os.path.abspath(model).endswith('.h5'):
                raise ValueError("Model names of the types %s must have the .h5 extension." % ['FFNN', 'CNN', 'LSTM', 'Transformer'])
        strategy = args.strategy
        model_list = args.models
        architecture_list = args.architectures
    elif args.models is not None or args.architectures is not None:
        raise ValueError("It is only allowed to use the --models and --architectures with the Ensemble architecture.")

    print("Loading Model...")
    # There are some problems regarding the loading of models on multiple GPU's.
    # gpu_count = len(tf.config.list_physical_devices('GPU'))
    # if gpu_count > 1:
    #     strat = tf.distribute.MirroredStrategy()
    #     with strat.scope():
    #         model = load_model()
    # else:
    #     model = load_model()
    model = load_model()
    print("Model Loaded.")

    # the program was started as in benchmark mode.
    if args.download_dataset is not None:
        benchmark(args, model)
    # the program was started in single_line mode.
    elif args.ciphertext is not None or args.file is not None:
        predict_single_line(args, model)
    # the program was started in prediction mode.
    else:
        evaluate(args, model)
