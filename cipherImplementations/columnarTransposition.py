import numpy as np
from cipherImplementations.cipher import Cipher
import random


class ColumnarTransposition(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number, fill_blocks):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number
        self.fill_blocks = fill_blocks

    def generate_random_key(self, length):
        if length is None or length <= 0:
            raise ValueError('The length of a key must be greater than 0 and must not be None.')
        if not isinstance(length, int):
            raise ValueError('Length must be of type integer.')
        key = list(range(length))
        random.shuffle(key)
        return key

    def encrypt(self, plaintext, key):
        ciphertext = []
        if self.fill_blocks:
            while len(plaintext) % len(key) != 0:
                if not isinstance(plaintext, list):
                    plaintext = list(plaintext)
                plaintext.append(self.alphabet.index(b'x'))
        for start in range(0, len(key)):
            position = key.index(start)
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
        plaintext = [b'']*len(ciphertext)
        i = 0
        for start in range(0, len(key)):
            position = key.index(start)
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