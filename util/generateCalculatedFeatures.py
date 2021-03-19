from pathlib import Path

import numpy as np
import argparse
import sys
import time
import os
import functools
from datetime import datetime
# This environ variable must be set before all tensorflow imports!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
sys.path.append("../")
import cipherTypeDetection.config as config
from cipherTypeDetection.textLine2CipherStatisticsDataset import TextLine2CipherStatisticsDataset
tf.debugging.set_log_device_placement(enabled=False)
# always flush after print as some architectures like RF need very long time before printing anything.
print = functools.partial(print, flush=True)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CANN Ciphertype Detection Neuronal Network Training Data Generation Script', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size for training.')
    parser.add_argument('--dataset_size', default=16000, type=int,
                        help='Dataset size per fit. This argument should be dividable \n'
                             'by the amount of --ciphers.')
    parser.add_argument('--dataset_workers', default=1, type=int,
                        help='The number of parallel workers for reading the \ninput files.')
    parser.add_argument('--input_directory', default='../data/gutenberg_en', type=str,
                        help='Input directory of the plaintexts.')
    parser.add_argument('--download_dataset', default=True, type=str2bool,
                        help='Download the dataset automatically.')
    parser.add_argument('--save_directory', default='../data/generated',
                        help='Directory for saving generated models. \n'
                             'When interrupting, the current model is \n'
                             'saved as interrupted_...')
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
    parser.add_argument('--max_iter', default=100000000, type=int,
                        help='the maximal number of iterations before stopping training.')
    parser.add_argument('--min_len', default=50, type=int,
                        help='The minimum length of a plaintext to be encrypted in testing. \n'
                             'If this argument is set to -1 no lower limit is used.')
    parser.add_argument('--max_len', default=-1, type=int,
                        help='The maximum length of a plaintext to be encrypted in training. \n'
                             'If this argument is set to -1 no upper limit is used.')

    args = parser.parse_args()
    for arg in vars(args):
        print("{:23s}= {:s}".format(arg, str(getattr(args, arg))))
    args.input_directory = os.path.abspath(args.input_directory)
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
    config.CIPHER_TYPES = cipher_types
    if args.dataset_size * args.dataset_workers > args.max_iter:
        print("ERROR: --dataset_size * --dataset_workers must not be bigger than --max_iter. "
              "In this case it was %d > %d" % (args.dataset_size * args.dataset_workers, args.max_iter), file=sys.stderr)
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

    ds = TextLine2CipherStatisticsDataset(plaintext_files, cipher_types, args.dataset_size, args.min_len, args.max_len,
                                          args.keep_unknown_symbols, args.dataset_workers, generate_test_data=True)
    if args.dataset_size % ds.key_lengths_count != 0:
        print("WARNING: the --dataset_size parameter must be dividable by the amount of --ciphers  and the length configured "
              "KEY_LENGTHS in config.py. The current key_lengths_count is %d" % ds.key_lengths_count, file=sys.stderr)
    print("Datasets loaded.\n")

    print('Calculating Features...')
    start_time = time.time()
    cntr = 0
    iterations = 0
    epoch = 0
    val_data = None
    val_labels = None
    run = None
    run1 = None
    ciphertexts = None
    ciphertexts1 = None
    processes = []
    new_run = [[], []]
    np.set_printoptions(threshold=np.inf)
    file_name_cntr = 1
    fd = None
    while ds.iteration < args.max_iter:
        if run1 is None:
            epoch = 0
            processes, run1, ciphertexts1 = ds.__next__()
        if run is None:
            fd = os.open(os.path.join(args.save_directory, str(file_name_cntr) + '.txt'), os.O_WRONLY | os.O_CREAT)
            for process in processes:
                process.join()
            run = run1
            ciphertexts = ciphertexts1
            ds.iteration += ds.batch_size * ds.dataset_workers
            if ds.iteration < args.max_iter:
                epoch = ds.epoch
                processes, run1, ciphertexts1 = ds.__next__()
        for i in range(len(run)):
            batch, labels = run[i]
            batch_ciphertexts = ciphertexts[i]
            cntr += 1
            iterations = args.dataset_size * cntr

            for j in range(len(labels)):
                batch_arr = bytearray()
                for b in batch[j]:
                    batch_arr += b'%f,' % b
                batch_arr = bytes(batch_arr)[:-1]
                batch_ciphers = bytearray()
                for c in batch_ciphertexts[j]:
                    batch_ciphers += b'%d,' % c
                batch_ciphers = bytes(batch_ciphers)[:-1]
                os.write(fd, b"%d %s %s\n" % (labels[j], batch_arr, batch_ciphers))

            if epoch > 0:
                epoch = iterations // ((ds.iteration + ds.batch_size * ds.dataset_workers) // ds.epoch)
            print("Epoch: %d, Iteration: %d" % (epoch, iterations))
            if iterations >= args.max_iter:
                break
        if ds.iteration >= args.max_iter:
            break
        run = None
        os.close(fd)
        file_name_cntr += 1
    for process in processes:
        if process.is_alive():
            process.terminate()

    elapsed_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)
    print('Finished data generation in %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.\n' % (
        elapsed_time.days, elapsed_time.seconds // 3600, (elapsed_time.seconds // 60) % 60, elapsed_time.seconds % 60, iterations, epoch))
