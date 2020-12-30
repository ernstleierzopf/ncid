import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword


class Variant(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        return generate_random_keyword(self.alphabet, length)

    def encrypt(self, plaintext, key):
        ciphertext = []
        for i in range(len(plaintext)):
            ciphertext.append((plaintext[i] + len(self.alphabet) - key[i % len(key)]) % len(self.alphabet))
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        for i in range(len(ciphertext)):
            plaintext.append((ciphertext[i] - len(self.alphabet) + key[i % len(key)]) % len(self.alphabet))
        return np.array(plaintext)
