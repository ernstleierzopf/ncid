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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CANN Ciphertype Detection Neuronal Network Evaluation Script', formatter_class=argparse.RawTextHelpFormatter)
    sp = parser.add_subparsers()
    bench_parser = sp.add_parser('benchmark', help='Use this argument to create ciphertexts on the fly, \nlike in '
                                    'training mode, and evaluate them with the model. \nThis option is optimized for large '
                                    'throughput to test the model.')
    eval_parser = sp.add_parser('evaluate', help='Use this argument to evaluate single files or directories.')
    single_line_parser = sp.add_parser('single_line', help='Use this argument to evaluate a single line of ciphertext.')

    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for training.')
    parser.add_argument('--max_iter', default=1000000, type=int,
                        help='the maximal number of iterations before stopping evaluation.')
    parser.add_argument('--model', default='weights/model.h5', type=str,
                        help='Name of the model file. The file must have the .h5 extension.')

    bench_parser.add_argument('--download_dataset', default=True, type=str2bool)
    bench_parser.add_argument('--dataset_workers', default=1, type=int)
    bench_parser.add_argument('--plaintext_folder', default='../data/gutenberg_en', type=str)
    bench_parser.add_argument('--ciphers', default='mtc3', type=str)
    bench_parser.add_argument('--keep_unknown_symbols', default=False, type=str2bool)
    bench_parser.add_argument('--min_text_len', default=50, type=int)
    bench_parser.add_argument('--max_text_len', default=-1, type=int)

    bench_group = parser.add_argument_group('benchmark')
    bench_group.add_argument('--download_dataset', help='Download the dataset automatically.')
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

    single_line_parser.add_argument('ciphertext', type=str)

    single_line_group = parser.add_argument_group('single_line')
    single_line_group.add_argument('ciphertext', help='A single line of ciphertext to be evaluated by the model.')

    args = parser.parse_args()
