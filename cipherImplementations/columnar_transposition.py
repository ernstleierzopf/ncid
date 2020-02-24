import numpy as np
from cipherImplementations.cipher import Cipher
import sys

sys.path.append("../../../")
from util import text_utils

class Columnar_transposition(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def encrypt(self, plaintext, key):
        key = self.indices(key)
        ciphertext = []
        for start in range(0, len(key)):
            position = text_utils.num_index_of(key, start)
            while position < len(plaintext):
                p = plaintext[position]
                if (p > len(self.alphabet)):
                    ciphertext.append(self.unknown_symbol_number)
                    position = position + len(key)
                    continue
                ciphertext.append(p)
                position = position + len(key)
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        key = self.indices(key)
        plaintext = [b'']*len(ciphertext)
        i = 0
        for start in range(0, len(key)):
            position = text_utils.num_index_of(key, start)
            while position < len(plaintext):
                c = ciphertext[i]
                i += 1
                if (c > len(self.alphabet)):
                    plaintext[position] = self.unknown_symbol_number
                    position = position + len(key)
                    continue
                plaintext[position] = c
                position = position + len(key)
        return np.array(plaintext)

    def indices(self, word):
        t1 = [(word[i], i) for i in range(len(word))]
        t2 = [(k[1], i) for i, k in enumerate(sorted(t1))]
        return [q[1] for q in sorted(t2)]