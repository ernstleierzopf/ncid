import argparse
import os
from pathlib import Path
import sys
import cipherImplementations as cipherImpl

sys.path.append("../../../")
from util import file_utils, text_utils

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def encrypt_file_with_all_cipher_types(filename, save_folder, cipher_types, append_key, keep_unknown_symbols, min_line_length, max_line_length):
    plaintexts = file_utils.read_txt_list_from_file(filename)
    for cipher_type in cipher_types:
        path = os.path.join(save_folder, cipher_type)
        if not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)
        index = cipherImpl.Cipher = cipherImpl.CIPHER_TYPES.index(cipher_type)
        if index > -1:
            cipher = cipherImpl.CIPHER_IMPLEMENTATIONS[index]
            key_length = cipherImpl.KEY_LENGTHS[index]
            ciphertexts = []
            keys = []
            for plaintext in plaintexts:
                if (not min_line_length is None and len(plaintext) < min_line_length) or (not max_line_length is None and len(plaintext) > max_line_length):
                    continue
                plaintext = cipher.filter(plaintext, keep_unknown_symbols)
                if plaintext == b'':
                    continue
                key = cipher.generate_random_key(key_length)
                keys.append(key)
                plaintext_numberspace = text_utils.map_text_into_numberspace(plaintext, cipher.alphabet, cipher.unknown_symbol_number)
                if isinstance(key, bytes):
                    key = text_utils.map_text_into_numberspace(key, cipher.alphabet, cipher.unknown_symbol_number)

                ciphertexts.append(text_utils.map_numbers_into_textspace(cipher.encrypt(plaintext_numberspace,key),
                    cipher.alphabet, cipher.unknown_symbol))

                #check if decryption works
                # c = cipher.encrypt(plaintext_numberspace, key)
                # c = text_utils.map_numbers_into_textspace(cipher.decrypt(c, key), cipher.alphabet, cipher.unknown_symbol)
                # if plaintext != c:
                #     print("plaintext: %s"%plaintext)
                #     print()
                #     print("ciphertext: %s"%c)
                #     print("error %d"%index)
            path = os.path.join(path, os.path.basename(filename))
            if append_key:
                file_utils.write_ciphertext_with_keys_to_file(path, ciphertexts, keys)
            else:
                file_utils.write_txt_list_to_file(path, ciphertexts)
        else:
            print('Cipher \'%s\' does not exist!'%cipher_type, sys.stderr)
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CANN Ciphertext Generator Script')
    parser.add_argument('--input_folder', default='../../gutenberg_test', type=str,
                        help='Input folder of the plaintexts.')
    parser.add_argument('--dataset_workers', default=4, type=str,
                        help='The number of parallel workers for reading the input files.')
    parser.add_argument('--save_folder', default='../../mtc3_cipher_id',
                        help='Directory for saving generated ciphertexts.'\
                             'For every cipher type a new subdirectory with'\
                             ' it\'s name is created.')
    parser.add_argument('--ciphers', default='mtc3', type=str,
                        help='A comma seperated list of the ciphers to be created. '\
                             'Be careful to not use spaces or use \' to define the string.'
                             'Possible values are:\n'
                             '- mtc3 (contains the ciphers Monoalphabetic Substitution, Vigenere, '\
                             'Columnar Transposition, Plaifair and Hill)\n'\
                             '- aca (contains all currently implemented ciphers from https://www.cryptogram.org/resource-area/cipher-types/)\n'\
                             '- monoalphabetic_substitution\n'\
                             '- vigenere'\
                             '- columnar_transposition'\
                             '- playfair'\
                             '- hill')
    parser.add_argument('--append_key', default=False, type=str2bool,
                        help='Append the encryption key at the end of every line.')
    parser.add_argument('--keep_unknown_symbols', default=True, type=str2bool,
                        help='Keep unknown symbols in the plaintexts. Known symbols are defined'\
                             'in the alphabet of the cipher.')
    parser.add_argument('--min_line_length', default=None, type=int,
                        help='Defines the minimal number of characters in a line to be chosen.'\
                             'This applies before spaces and other non-encryptable characters are filtered.'\
                             'If this parameter is None, no minimal line length will be checked.')
    parser.add_argument('--max_line_length', default=None, type=int,
                        help='Defines the maximal number of characters in a sentence to be chosen.'\
                             'This applies before spaces and other non-encryptable characters are filtered.'\
                             'If this parameter is None, no maximal line length will be checked.')
    args = parser.parse_args()
    args.input_folder = os.path.abspath(args.input_folder)
    args.save_folder = os.path.abspath(args.save_folder)
    args.ciphers = args.ciphers.lower()
    cipher_types = args.ciphers.split(',')
    if cipherImpl.MTC3 in cipher_types:
        del cipher_types[cipher_types.index(cipherImpl.MTC3)]
        cipher_types.append(cipherImpl.CIPHER_TYPES[0])
        cipher_types.append(cipherImpl.CIPHER_TYPES[1])
        cipher_types.append(cipherImpl.CIPHER_TYPES[2])
        cipher_types.append(cipherImpl.CIPHER_TYPES[3])
        cipher_types.append(cipherImpl.CIPHER_TYPES[4])
    if not os.path.exists(args.save_folder):
        Path(args.save_folder).mkdir(parents=True, exist_ok=True)

    #print all arguments for debugging..
    for arg in vars(args):
        print("%s = %s"%(arg, vars(args)[arg]))

    total_file_count = 0
    for root, dirs, files in os.walk(args.input_folder):
        total_file_count += len(files)

    dir = os.listdir(args.input_folder)
    file_counter = 0
    for name in dir:
        if os.path.isfile(os.path.join(args.input_folder, name)):
            file_counter += 1
            encrypt_file_with_all_cipher_types(os.path.join(args.input_folder, name), args.save_folder, cipher_types,
                args.append_key, args.keep_unknown_symbols, args.min_line_length, args.max_line_length)
            file_utils.print_progress('Encrypting files: [', file_counter, total_file_count, name)