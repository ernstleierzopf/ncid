from cipherImplementations.cipher import Cipher, generate_random_keyword
import numpy as np
import copy


def split_text(text, length):
    table = []
    for k in range(0, len(text), length):
        table.append(list(text[k:k+length]))
    return table


class Cadenus(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        if length is None or length <= 0 or length > len(self.alphabet.replace(b'w', b'')):
            raise ValueError('The length of a key must be greater than 0 and smaller than the size of the alphabet.')
        if not isinstance(length, int):
            raise ValueError('Length must be of type integer.')
        return [generate_random_keyword(self.alphabet, length, unique=True), b'azyxvutsrqponmlkjihgfedcb']

    def encrypt(self, plaintext, key):
        if len(plaintext) % 25 != 0 or len(plaintext) / 25 != len(key[0]):
            raise ValueError('The length of the keyword must be the same with the plaintext length divided by 25. The length of the '
                             'plaintext must be divisible by 25.')
        ciphertext = []
        pt_table = split_text(plaintext, len(key[0]))
        keyword = key[0]
        keyalphabet = key[1]
        ordered_keyword = copy.copy(keyword)
        ordered_keyword.sort()
        for _ in range(len(pt_table)):
            for c in ordered_keyword:
                r = c
                if r == 22:  # c = 'w' -> c = 'v'
                    r = 21
                ciphertext.append(pt_table[np.where(keyalphabet == r)[0][0]][np.where(keyword == c)[0][0]])
            pt_table.append(pt_table[0])
            pt_table = pt_table[1:]
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        if len(ciphertext) % 25 != 0 or len(ciphertext) / 25 != len(key[0]):
            raise ValueError('The length of the keyword must be the same with the plaintext length divided by 25. The length of the '
                             'plaintext must be divisible by 25.')
        plaintext = []
        ct_table = split_text(ciphertext, len(key[0]))
        pt_table = [[0]*len(key[0]) for _ in range(len(ct_table))]
        keyword = key[0]
        keyalphabet = key[1]
        ordered_keyword = copy.copy(keyword)
        ordered_keyword.sort()
        for i in range(len(pt_table)):
            for c in ordered_keyword:
                r = c
                if r == 22:  # c = 'w' -> c = 'v'
                    r = 21
                pt_table[(np.where(keyalphabet == r)[0][0] + i) % len(pt_table)][np.where(keyword == c)[0][0]] = ct_table[i][
                    np.where(ordered_keyword == c)[0][0]]
        for i in range(len(pt_table)):
            plaintext += pt_table[i]
        return np.array(plaintext)
