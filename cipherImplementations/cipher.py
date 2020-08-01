import random
import sys
import numpy as np
from cipherTypeDetection.dictionary import *
from collections import OrderedDict

sys.path.append("../../../")
from util.textUtils import remove_unknown_symbols, map_text_into_numberspace


def generate_random_list_of_unique_digits(length):
    if length is None or length <= 0:
        raise ValueError('The length of a key must be greater than 0 and must not be None.')
    if not isinstance(length, int):
        raise ValueError('Length must be of type integer.')
    key = list(range(length))
    random.shuffle(key)
    return np.array(key)


def generate_random_keyword(alphabet, length, unique=False):
    if length is None or length <= 0:
        raise ValueError('The length of a key must be greater than 0 and must not be None.')
    if not isinstance(length, int):
        raise ValueError('Length must be of type integer.')
    if USE_DICTIONARY:
        raise Exception('Dictionary key generation not implemented yet..')

    key = bytearray()
    alphabet2 = b'' + alphabet
    for _ in range(length):
        char = alphabet2[random.randint(0, len(alphabet2) - 1)]
        key.append(char)
        if unique:
            alphabet2 = alphabet2.replace(bytes([char]), b'')
    return bytes(key)


def generate_keyword_alphabet(alphabet, keyword, shift_randomly=False, vertical=False, indexed_kw_transposition=False, second_index_kw=None):
    key = []
    for char in keyword:
        key.append(alphabet.index(char))
    # remove duplicates
    key = OrderedDict.fromkeys(key)

    alph_out = bytearray()
    for i in key:
        alph_out.append(alphabet[i])

    for i in range(len(alphabet)):
        if i not in key:
            alph_out.append(alphabet[i])
    alph_out = bytes(alph_out)

    if shift_randomly:
        offset = random.randint(1, 10)
        new_key = bytearray()
        for i in range(len(alph_out)):
            new_key.append(alph_out[(i + offset) % len(alph_out)])
        return bytes(new_key)

    if vertical:
        new_key = [b''] * len(alph_out)
        for i, c in enumerate(alph_out):
            new_key[(3 * i) % len(alph_out)] = bytes([c])
        alph_out = b''
        for c in new_key:
            alph_out += c
        return alph_out

    if indexed_kw_transposition:
        t = map_text_into_numberspace(keyword, alphabet, 90)
        indizes = np.argsort(map_text_into_numberspace(keyword, alphabet, 90))
        if second_index_kw is not None:
            indizes = np.argsort(map_text_into_numberspace(second_index_kw, alphabet, 90))
        alph = bytearray()
        for start in indizes:
            position = start
            while position < len(alph_out):
                p = alph_out[position]
                alph.append(p)
                position = position + len(indizes)
        return bytes(alph)
    return alph_out


class Cipher:
    """This is the interface of the cipher implementations."""
    def generate_random_key(self, length):
        raise Exception('Interface method called')

    def encrypt(self, plaintext, key):
        raise Exception('Interface method called')

    def decrypt(self, ciphertext, key):
        raise Exception('Interface method called')

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower()
        if not keep_unknown_symbols:
            return remove_unknown_symbols(plaintext, self.alphabet)
        return plaintext