import numpy as np
from cipherImplementations.cipher import Cipher
from util.textUtils import num_index_of
import random


def indices(word):
    t1 = [(word[i], i) for i in range(len(word))]
    t2 = [(k[1], i) for i, k in enumerate(sorted(t1))]
    return [q[1] for q in sorted(t2)]


class ColumnarTransposition(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        if length is None or length <= 0:
            raise ValueError('The length of a key must be greater than 0 and must not be None.')
        if not isinstance(length, int):
            raise ValueError('Length must be of type integer.')
        key = list(range(length))
        random.shuffle(key)
        return key

    def encrypt(self, plaintext, key):
        key = indices(key)
        ciphertext = []
        plaintext = list(plaintext)
        while len(plaintext) % len(key) != 0:
            plaintext.append(self.alphabet.index(b'x'))
        for start in range(0, len(key)):
            position = num_index_of(key, start)
            while position < len(plaintext):
                p = plaintext[position]
                if p > len(self.alphabet):
                    ciphertext.append(self.unknown_symbol_number)
                    position = position + len(key)
                    continue
                ciphertext.append(p)
                position = position + len(key)
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        key = indices(key)
        plaintext = [b'']*len(ciphertext)
        i = 0
        for start in range(0, len(key)):
            position = num_index_of(key, start)
            while position < len(plaintext):
                c = ciphertext[i]
                i += 1
                if c > len(self.alphabet):
                    plaintext[position] = self.unknown_symbol_number
                    position = position + len(key)
                    continue
                plaintext[position] = c
                position = position + len(key)
        return np.array(plaintext)