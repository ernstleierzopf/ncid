import argparse
import os
from pathlib import Path
import sys
sys.path.append("../")
from cipherTypeDetection import config as config
from cipherTypeDetection.textLine2CipherStatisticsDataset import encrypt
from cipherImplementations.cipher import OUTPUT_ALPHABET, UNKNOWN_SYMBOL
from util.utils import read_txt_list_from_file, write_ciphertexts_with_keys_to_file, map_numbers_into_textspace,\
    write_txt_list_to_file, print_progress


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def encrypt_file_with_all_cipher_types(filename, save_folder, cipher_types_, append_key, keep_unknown_symbols, min_text_len, max_text_len):
    plaintexts = read_txt_list_from_file(filename)
    for cipher_type in cipher_types_:
        index = config.Cipher = config.CIPHER_TYPES.index(cipher_type)
        if index > -1:
            print('Encrypting File: %s, Cipher: %s' % (filename, cipher_type))
            cipher = config.CIPHER_IMPLEMENTATIONS[index]
            key_length = config.KEY_LENGTHS[index][0]
            ciphertexts = []
            keys = []
            for p in plaintexts:
                if len(cipher.filter(p, keep_unknown_symbols)) < min_text_len:
                    continue
                ciphertext, key = encrypt(p, index, key_length, keep_unknown_symbols, True)
                ciphertexts.append(map_numbers_into_textspace(ciphertext, OUTPUT_ALPHABET, UNKNOWN_SYMBOL))
                keys.append(key)

                # check if decryption works
                # c = cipher.encrypt(plaintext_numberspace, key)
                # c = text_utils.map_numbers_into_textspace(cipher.decrypt(c, key), config.ALPHABET, config.UNKNOWN_SYMBOL)
                # if plaintext != c:
                #     print("plaintext: %s"%plaintext)
                #     print()
                #     print("ciphertext: %s"%c)
                #     print("error %d"%index)
            path = os.path.join(save_folder, os.path.basename(filename).split('.txt')[0] + '-' + cipher_type + '-minLen' +
                                str(min_text_len) + '-maxLen' + str(max_text_len) + '-keyLen' + str(key_length) + '.txt')
            key_path = os.path.join(save_folder, 'keys', os.path.basename(filename).split('.txt')[0] + '-' + cipher_type + '.txt')
            if append_key:
                write_ciphertexts_with_keys_to_file(path, ciphertexts, keys)
            else:
                write_txt_list_to_file(path, ciphertexts)
            write_txt_list_to_file(key_path, keys)
        else:
            print('Cipher \'%s\' does not exist!' % cipher_type, sys.stderr)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CANN Ciphertext Generator Script', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_folder', default='../gutenberg_en', type=str,
                        help='Input folder of the plaintexts.')
    parser.add_argument('--save_folder', default='../mtc3_cipher_id',
                        help='Directory for saving generated ciphertexts.\n'
                             'For every cipher type a new subdirectory with\n'
                             'it\'s name is created.')
    parser.add_argument('--ciphers', default='aca', type=str,
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
    parser.add_argument('--append_key', default=False, type=str2bool,
                        help='Append the encryption key at the end of every line.')
    parser.add_argument('--keep_unknown_symbols', default=False, type=str2bool,
                        help='Keep unknown symbols in the plaintexts. Known symbols are defined\n'
                             'in the alphabet of the cipher.')
    parser.add_argument('--min_text_len', default=50, type=int,
                        help='The minimum length of a plaintext to be encrypted in the evaluation process.\n'
                             'If this argument is set to -1 no lower limit is used.')
    parser.add_argument('--max_text_len', default=-1, type=int,
                        help='The maximum length of a plaintext to be encrypted in the evaluation process.\n'
                             'If this argument is set to -1 no upper limit is used.')
    parser.add_argument('--max_files_count', default=-1, type=int,
                        help='Define the amount of files to be encrypted by every cipher.\n'
                             'If set to -1 all files are encrypted by every cipher')
    args = parser.parse_args()
    args.input_folder = os.path.abspath(args.input_folder)
    args.save_folder = os.path.abspath(args.save_folder)
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
    if not os.path.exists(args.save_folder) or not os.path.exists(args.save_folder+'/keys'):
        Path(args.save_folder).mkdir(parents=True, exist_ok=True)
        Path(args.save_folder+'/keys').mkdir(parents=True, exist_ok=True)

    # print all arguments for debugging..
    for arg in vars(args):
        print("%s = %s" % (arg, vars(args)[arg]))

    total_file_count = 0
    for root, dirs, files in os.walk(args.input_folder):
        total_file_count += len(files)

    if total_file_count > args.max_files_count > -1:
        total_file_count = args.max_files_count

    dir_name = os.listdir(args.input_folder)
    file_counter = 0
    for name in dir_name:
        if os.path.isfile(os.path.join(args.input_folder, name)):
            file_counter += 1
            encrypt_file_with_all_cipher_types(os.path.join(args.input_folder, name), args.save_folder, cipher_types, args.append_key,
                                               args.keep_unknown_symbols, args.min_text_len, args.max_text_len)
            print_progress('Encrypting files: [', file_counter, total_file_count)
            if file_counter == total_file_count:
                break
