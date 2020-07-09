from pathlib import Path

import argparse
import sys
import os
from datetime import datetime
# This environ variable must be set before all tensorflow imports!
from util import textUtils, fileUtils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
sys.path.append("../")
import cipherTypeDetection.config as config
from cipherTypeDetection.textLine2CipherStatisticsDataset import TextLine2CipherStatisticsDataset, calculate_statistics
tf.debugging.set_log_device_placement(enabled=False)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def benchmark(args, model):
    args.plaintext_folder = os.path.abspath(args.plaintext_folder)
    if args.dataset_size * args.dataset_workers > args.max_iter:
        print("ERROR: --dataset_size * --dataset_workers must not be bigger than --max_iter. In this case it was %d > %d" % (
            args.dataset_size * args.dataset_workers, args.max_iter), file=sys.stderr)
        exit(1)
    if args.download_dataset and not os.path.exists(args.plaintext_folder) and args.plaintext_folder == os.path.abspath(
            '../data/gutenberg_en'):
        print("Downloading Datsets...")
        tfds.download.add_checksums_dir('../data/checksums/')
        download_manager = tfds.download.download_manager.DownloadManager(download_dir='../data/', extract_dir=args.plaintext_folder)
        download_manager.download_and_extract(
            'https://drive.google.com/uc?id=1bF5sSVjxTxa3DB-P5wxn87nxWndRhK_V&export=download')
        path = os.path.join(args.plaintext_folder, 'ZIP.ucid_1bF5sSVjxTx-P5wxn87nxWn_V_export_downloadR9Cwhunev5CvJ-ic__'
                                                   'HawxhTtGOlSdcCrro4fxfEI8A', os.path.basename(args.plaintext_folder))
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
    dir_nam = os.listdir(args.plaintext_folder)
    for name in dir_nam:
        path = os.path.join(args.plaintext_folder, name)
        if os.path.isfile(path):
            plaintext_files.append(path)
    dataset = TextLine2CipherStatisticsDataset(plaintext_files, cipher_types, args.dataset_size, args.min_text_len, args.max_text_len,
                                               args.keep_unknown_symbols, args.dataset_workers)
    if args.dataset_size % dataset.key_lengths_count != 0:
        print("WARNING: the --dataset_size parameter must be dividable by the amount of --ciphers  and the length configured KEY_LENGTHS in"
              " config.py. The current key_lengths_count is %d" % dataset.key_lengths_count, file=sys.stderr)
    print("Datasets loaded.\n")

    print("Shuffling data...")
    dataset = dataset.shuffle(50000, seed=42, reshuffle_each_iteration=False)
    print("Data shuffled.\n")

    print('Evaluating model...')
    import time
    start_time = time.time()
    cntr = 0
    iteration = 0
    epoch = 0
    results = []
    while dataset.iteration < args.max_iter:
        for run in dataset:
            for batch, labels in run:
                results.append(model.evaluate(batch, labels, batch_size=args.batch_size))
                cntr += 1
                iteration = args.dataset_size * cntr
                epoch = dataset.epoch
                if epoch > 0:
                    epoch = iteration // (dataset.iteration // dataset.epoch)
                print("Epoch: %d, Iteration: %d" % (epoch, iteration))
                if iteration >= args.max_iter:
                    break
            if dataset.iteration >= args.max_iter:
                break
        if dataset.iteration >= args.max_iter:
            break
    elapsed_evaluation_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)
    print('Finished evaluation in %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.\n' % (
        elapsed_evaluation_time.days, elapsed_evaluation_time.seconds // 3600, (elapsed_evaluation_time.seconds // 60) % 60,
        elapsed_evaluation_time.seconds % 60, iteration, epoch))

    avg_loss = 0
    avg_acc = 0
    for loss, acc in results:
        avg_loss += loss
        avg_acc += acc
    avg_loss = avg_loss / len(results)
    avg_acc = avg_acc / len(results)

    print("Average evaluation results: loss: %f, accuracy: %f\n" % (avg_loss, avg_acc))


def evaluate(args, model):
    results_list = []
    dir_name = os.listdir(args.ciphertext_folder)
    dir_name.sort()
    cntr = 0
    iterations = 0
    basename = ''
    for name in dir_name:
        if iterations > args.max_iter:
            break
        path = os.path.join(args.ciphertext_folder, name)
        if os.path.isfile(path):
            if iterations > args.max_iter:
                break
            batch = []
            label = config.CIPHER_TYPES.index(os.path.basename(path).split('-')[1])
            if os.path.basename(path).split('-')[0] != basename:
                print()
            basename = os.path.basename(path).split('-')[0]
            with open(path, "rb") as fd:
                for line in fd.readlines():
                    # remove newline
                    line = line[:-1]
                    ciphertext = textUtils.map_text_into_numberspace(line, config.ALPHABET, config.UNKNOWN_SYMBOL_NUMBER)
                    statistics = calculate_statistics(ciphertext)
                    batch.append(statistics)
                    iterations += 1
                    if iterations == args.max_iter:
                        break
            result = model.evaluate(tf.convert_to_tensor(batch), tf.convert_to_tensor([label]*len(batch)), args.batch_size, verbose=0)
            results_list.append(result)
            cntr += 1
            if args.evaluation_mode == 'per_file':
                print("%s (%d lines) test_loss: %f, test_accuracy: %f (progress: %d%%)" % (
                    os.path.basename(path), len(batch), result[0], result[1], max(
                        int(cntr / len(dir_name) * 100), int(iterations / args.max_iter) * 100)))
            else:
                fileUtils.print_progress("Evaluating files", cntr, len(dir_name), factor=5)

    avg_test_loss = 0
    avg_test_acc = 0
    for loss, acc in results_list:
        avg_test_loss += loss
        avg_test_acc += acc
    avg_test_loss = avg_test_loss / len(results_list)
    avg_test_acc = avg_test_acc / len(results_list)
    print("\n\nAverage evaluation results from %d iterations: avg_test_loss=%f, avg_test_acc=%f" % (
        iterations, avg_test_loss, avg_test_acc))


def predict_single_line(args, model):
    cipher_id_result = ''
    ciphertexts = []
    if args.ciphertext is not None:
        ciphertexts.append(bytes(args.ciphertext, 'ascii'))
    else:
        ciphertexts = open(args.file, 'rb')

    print()
    for line in ciphertexts:
        # remove newline
        line = line[:-1]
        print(line)
        ciphertext = textUtils.map_text_into_numberspace(line, config.ALPHABET, config.UNKNOWN_SYMBOL_NUMBER)
        statistics = calculate_statistics(ciphertext)
        result = model.predict(tf.convert_to_tensor([statistics]), args.batch_size, verbose=0)
        if args.verbose:
            for cipher in args.ciphers:
                print("{:23s} {:f}%".format(cipher, result[0].tolist()[config.CIPHER_TYPES.index(cipher)]*100))
            result_list = result[0].tolist()
            max_val = max(result_list)
            cipher = config.CIPHER_TYPES[result_list.index(max_val)]
        else:
            result_list = result[0].tolist()
            max_val = max(result_list)
            cipher = config.CIPHER_TYPES[result_list.index(max_val)]
            print("{:s} {:f}%".format(cipher, max_val * 100))
        print()
        cipher_id_result += cipher[0].upper()

    if args.file is not None:
        ciphertexts.close()
    # print(cipher_id_result)
    # print('C: %d' %cipher_id_result.count('C'))
    # print('H: %d' % cipher_id_result.count('H'))
    # print('P: %d' % cipher_id_result.count('P'))
    # print('S: %d' % cipher_id_result.count('S'))
    # print('V: %d' % cipher_id_result.count('V'))
    # solution = 'SCPHHCVHSSSHHVVPSSVPVCVVHPHCCVPHSPVPPPSHHCSHSVPPSSHVCVSCSCSSCVVSVHSHSSCCPVHVPHPPSPSHCVHCCSSVHHCHPSVSSVCHVCPSVPHVVPPVCHCPCSVCHVCVCPPPSCVPHPVVHCCSVHHHSPCPHCCVHHPSCSPVSCCHVCSPHHHSCCPSPPCVVVVHVCSCSVVHHHPHPCVVHPPVVCSVHCSHHVSVPVCPSHSSVHPPCCHCSSVVCPHSCCPCHCCHVHCHVVVSCSPSVPVCSCCPSSSVHCPSPHVVPCHHSPPHVCHSPPCHVCHPCCPCCPSPSSSVHVSSSSHPVCSCPPCHPSPCVCPHPSSSCSHHCSPVSVPVSSPCVHHVPPCPPPHSCHPCSHPHPPCVSCCSHCVHVHCPCHCSHPVCVVSPPPSCVHPPHHHVPSSVVCCPSPVHSCHHSPVSHHHVSCHHPSCHPVPSHCPVHVCHVVPHVVHSSHVVCPVPPSSCVPHVSPPCCSCHVCCS'
    #
    # for i, c in enumerate(cipher_id_result):
    #     if c != solution[i]:
    #         print('solution: %s, prediction: %s, position: %d' % (solution[i], c, i))


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
    parser.add_argument('--max_iter', default=1000000, type=int,
                        help='the maximal number of iterations before stopping evaluation.')
    parser.add_argument('--model', default='./weights/model.h5', type=str,
                        help='Name of the model file. The file must have the .h5 extension.')
    parser.add_argument('--ciphers', '--ciphers', default='mtc3', type=str,
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
                             '- hill\n'
                             '- amsco\n'
                             '- autokey')

    bench_parser.add_argument('--download_dataset', default=True, type=str2bool)
    bench_parser.add_argument('--dataset_size', default=16000, type=int)
    bench_parser.add_argument('--dataset_workers', default=1, type=int)
    bench_parser.add_argument('--plaintext_folder', default='../data/gutenberg_en', type=str)
    bench_parser.add_argument('--keep_unknown_symbols', default=False, type=str2bool)
    bench_parser.add_argument('--min_text_len', default=50, type=int)
    bench_parser.add_argument('--max_text_len', default=-1, type=int)

    bench_group = parser.add_argument_group('benchmark')
    bench_group.add_argument('--download_dataset', help='Download the dataset automatically.')
    bench_group.add_argument('--dataset_size', help='Dataset size per evaluation. This argument should be dividable \n'
                             'by the amount of --ciphers.')
    bench_group.add_argument('--dataset_workers', help='The number of parallel workers for reading the input files.')
    bench_group.add_argument('--plaintext_folder', help='Input folder of the plaintexts.')
    bench_group.add_argument('--keep_unknown_symbols', help='Keep unknown symbols in the plaintexts. Known \n'
                                                            'symbols are defined in the alphabet of the cipher.')
    bench_group.add_argument('--min_text_len', help='The minimum length of a plaintext to be encrypted in the evaluation process.\n'
                             'If this argument is set to -1 no lower limit is used.')
    bench_group.add_argument('--max_text_len', help='The maximum length of a plaintext to be encrypted in the evaluation process.\n'
                             'If this argument is set to -1 no upper limit is used.')

    eval_parser.add_argument('--evaluation_mode', nargs='?', choices=('summarized', 'per_file'), default='summarized', type=str)
    eval_parser.add_argument('--ciphertext_folder', default='../data/ciphertexts_gutenberg_en', type=str)

    eval_group = parser.add_argument_group('evaluate')
    eval_group.add_argument('--evaluation_mode',
                            help='- To create an single evaluation result over all iterated ciphertext files use the \'summarized\' option.\n'
                                 '  This option is to be preferred over the benchmark option, if the tests should be reproducable.\n'
                                 '- To create an evaluation for every file use \'per_file\' option. This mode allows the \n'
                                 '  calculation of the \n  - average value of the prediction \n'
                                 '  - lower quartile - value at the position of 25 percent of the sorted predictions\n'
                                 '  - median - value at the position of 50 percent of the sorted predictions\n'
                                 '  - upper quartile - value at the position of 75 percent of the sorted predictions\n'
                                 '  With these statistics an expert can classify a ciphertext document to a specific cipher.')
    eval_group.add_argument('--ciphertext_folder', help='Input folder of the ciphertext files.')

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
        exit(1)

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
    args.ciphers = cipher_types

    print("Loading Model...")
    gpu_count = len(tf.config.list_physical_devices('GPU'))
    if gpu_count > 1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = tf.keras.models.load_model(args.model)
    else:
        model = tf.keras.models.load_model(args.model)
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