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
import cipherTypeDetection.config as config
from cipherTypeDetection.textLine2CipherStatisticsDataset import TextLine2CipherStatisticsDataset
tf.debugging.set_log_device_placement(enabled=False)
import math

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def benchmark(args, model):
    args.plaintext_folder = os.path.abspath(args.plaintext_folder)
    args.ciphers = args.ciphers.lower()
    cipher_types = args.ciphers.split(',')
    if config.MTC3 in cipher_types:
        del cipher_types[cipher_types.index(config.MTC3)]
        cipher_types.append(config.CIPHER_TYPES[0])
        cipher_types.append(config.CIPHER_TYPES[1])
        cipher_types.append(config.CIPHER_TYPES[2])
        cipher_types.append(config.CIPHER_TYPES[3])
        cipher_types.append(config.CIPHER_TYPES[4])
    if args.dataset_size * args.dataset_workers > args.max_iter:
        print("ERROR: --dataset_size * --dataset_workers must not be bigger than --max_iter. "
              "In this case it was %d > %d" % (args.dataset_size * args.dataset_workers, args.max_iter),
            file=sys.stderr)
        exit(1)
    if args.download_dataset and not os.path.exists(args.input_folder) and args.input_folder == os.path.abspath(
            '../data/gutenberg_en'):
        print("Downloading Datsets...")
        tfds.download.add_checksums_dir('../data/checksums/')
        download_manager = tfds.download.download_manager.DownloadManager(download_dir='../data/',
            extract_dir=args.plaintext_folder)
        download_manager.download_and_extract(
            'https://drive.google.com/uc?id=1bF5sSVjxTxa3DB-P5wxn87nxWndRhK_V&export=download')
        path = os.path.join(args.plaintext_folder,
            'ZIP.ucid_1bF5sSVjxTx-P5wxn87nxWn_V_export_downloadR9Cwhunev5CvJ-ic__HawxhTtGOlSdcCrro4fxfEI8A.incomplete_25fe7c1666cb4a8fb06682d99df2c0df',
            os.path.basename(args.plaintext_folder))
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
    dir = os.listdir(args.plaintext_folder)
    for name in dir:
        path = os.path.join(args.plaintext_folder, name)
        if os.path.isfile(path):
            plaintext_files.append(path)
    dataset = TextLine2CipherStatisticsDataset(plaintext_files, cipher_types, args.dataset_size, args.min_text_len,
        args.max_text_len, args.keep_unknown_symbols, args.dataset_workers)
    if args.dataset_size % dataset.key_lengths_count != 0:
        print("WARNING: the --dataset_size parameter must be dividable by the amount of --ciphers "
              " and the length configured KEY_LENGTHS in config.py. The current key_lengths_count is %d" % dataset.key_lengths_count,
            file=sys.stderr)
    print("Datasets loaded.\n")

    print("Shuffling data...")
    train_dataset = dataset.shuffle(50000, seed=42, reshuffle_each_iteration=False)
    print("Data shuffled.\n")

    print('Evaluating model...')
    import time
    start_time = time.time()
    cntr = 0
    iter = 0
    epoch = 0
    results = []
    while dataset.iteration < args.max_iter:
        for run in dataset:
            for batch, labels in run:
                results.append(model.evaluate(batch, labels, batch_size=args.batch_size))
                cntr += 1
                iter = args.dataset_size * cntr
                epoch = dataset.epoch
                if epoch > 0:
                    epoch = iter // (dataset.iteration // dataset.epoch)
                print("Epoch: %d, Iteration: %d" % (epoch, iter))
                if iter >= args.max_iter:
                    break
            if dataset.iteration >= args.max_iter:
                break
        if dataset.iteration >= args.max_iter:
            break
    elapsed_evaluation_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)
    print('Finished evaluation in %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.\n' % (
    elapsed_evaluation_time.days, elapsed_evaluation_time.seconds // 3600, (elapsed_evaluation_time.seconds // 60) % 60,
    (elapsed_evaluation_time.seconds) % 60, iter, epoch))

    avg_loss = 0
    avg_acc = 0
    for loss, acc in results:
        avg_loss += loss
        avg_acc += acc
    avg_loss = avg_loss / len(results)
    avg_acc = avg_acc / len(results)

    print("Average evaluation results: loss: %f, accuracy: %f\n" % (avg_loss, avg_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CANN Ciphertype Detection Neuronal Network Evaluation Script', formatter_class=argparse.RawTextHelpFormatter)
    sp = parser.add_subparsers()
    bench_parser = sp.add_parser('benchmark', help='Use this argument to create ciphertexts on the fly, \nlike in '
                                    'training mode, and evaluate them with the model. \nThis option is optimized for large '
                                    'throughput to test the model.')
    eval_parser = sp.add_parser('evaluate', help='Use this argument to evaluate single files or directories.')
    single_line_parser = sp.add_parser('single_line', help='Use this argument to evaluate a single line of ciphertext.')

    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size for training.')
    parser.add_argument('--max_iter', default=1000000, type=int,
                        help='the maximal number of iterations before stopping evaluation.')
    parser.add_argument('--model', default='weights/model.h5', type=str,
                        help='Name of the model file. The file must have the .h5 extension.')

    bench_parser.add_argument('--download_dataset', default=True, type=str2bool)
    bench_parser.add_argument('--dataset_size', default=16000, type=int)
    bench_parser.add_argument('--dataset_workers', default=1, type=int)
    bench_parser.add_argument('--plaintext_folder', default='../data/gutenberg_en', type=str)
    bench_parser.add_argument('--ciphers', default='mtc3', type=str)
    bench_parser.add_argument('--keep_unknown_symbols', default=False, type=str2bool)
    bench_parser.add_argument('--min_text_len', default=50, type=int)
    bench_parser.add_argument('--max_text_len', default=-1, type=int)

    bench_group = parser.add_argument_group('benchmark')
    bench_group.add_argument('--download_dataset', help='Download the dataset automatically.')
    bench_group.add_argument('--dataset_size', help='Dataset size per evaluation. This argument should be dividable \n'
                             'by the amount of --ciphers.')
    bench_group.add_argument('--dataset_workers', help='The number of parallel workers for reading the input files.')
    bench_group.add_argument('--plaintext_folder', help='Input folder of the plaintexts.')
    bench_group.add_argument('--ciphers', help='A comma seperated list of the ciphers to be created.\n'
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
                             '- hill')
    bench_group.add_argument('--keep_unknown_symbols', help='Keep unknown symbols in the plaintexts. Known \n'
                                                            'symbols are defined in the alphabet of the cipher.')
    bench_group.add_argument('--min_text_len', help='The minimum length of a plaintext to be encrypted in the evaluation process.\n'
                             'If this argument is set to -1 no lower limit is used.')
    bench_group.add_argument('--max_text_len', help='The maximum length of a plaintext to be encrypted in the evaluation process.\n'
                             'If this argument is set to -1 no upper limit is used.')

    eval_parser.add_argument('--evaluation_mode', nargs='?', choices=('summarized', 'per_file'), default='summarized')
    eval_parser.add_argument('--ciphertext_folder', default='../data/ciphertexts_gutenberg_en', type=str)

    eval_group = parser.add_argument_group('evaluate')
    eval_group.add_argument('--evaluation_mode', help='- To create an single evaluation over all iterated ciphertext files use the \'summarized\' option.\n'
                                                      '  This option is to be preferred over the benchmark option, if the tests should be reproducable.\n'
                                                      '- To create an evaluation for every file use \'per_file\' option. This mode allows the \n'
                                                      '  calculation of the \n  - average value of the prediction \n'
                                                      '  - lower quartile - value at the position of 25 percent of the sorted predictions\n'
                                                      '  - median - value at the position of 50 percent of the sorted predictions\n'
                                                      '  - upper quartile - value at the position of 75 percent of the sorted predictions\n'
                                                      '  With these statistics an expert can classify a ciphertext document to a specific cipher.')
    eval_group.add_argument('--ciphertext_folder', help='Input folder of the ciphertext files.')

    single_line_parser.add_argument('--ciphertext', type=str)

    single_line_group = parser.add_argument_group('single_line')
    single_line_group.add_argument('--ciphertext', help='A single line of ciphertext to be evaluated by the model.')

    args = parser.parse_args()
    for arg in vars(args):
        print("{:23s}= {:s}".format(arg, str(getattr(args, arg))))
    m = os.path.splitext(args.model)
    if len(os.path.splitext(args.model)) != 2 or os.path.splitext(args.model)[1] != '.h5':
        print('ERROR: The model name must have the ".h5" extension!', file=sys.stderr)
        exit(1)

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
    if hasattr(args, 'download_dataset'):
        benchmark(args, model)
    # the program was started in evaluate mode.
    elif hasattr(args, 'evaluation_mode'):
        pass
    # the program was started in single_line mode.
    else:
        pass

