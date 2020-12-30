import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_list_of_unique_digits


class Gronsfeld(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        return generate_random_list_of_unique_digits(length)

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
