import random
import numpy as np
from cipherImplementations.cipher import Cipher


class Gronsfeld(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        if length is None or length <= 0:
            raise ValueError('The length of a key must be greater than 0 and must not be None.')
        if not isinstance(length, int):
            raise ValueError('Length must be of type integer.')
        key = []
        for i in range(length):
            key.append(random.randint(0, 9))
        return key

    def encrypt(self, plaintext, key):
        ciphertext = []
        for i, p in enumerate(plaintext):
            ciphertext.append((p + key[i % len(key)]) % len(self.alphabet))
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        for i, c in enumerate(ciphertext):
            plaintext.append((c - key[i % len(key)]) % len(self.alphabet))
        return np.array(plaintext)