import numpy as np
from cipherImplementations.cipher import Cipher
import sys

sys.path.append("../../../")
from util import text_utils

class Vigenere(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def encrypt(self, plaintext, key):
        keylength = len(key)
        ciphertext = []
        for position in range(0, len(plaintext)):
            p = plaintext[position]
            if (p > len(self.alphabet)):
                ciphertext.append(self.unknown_symbol_number)
                continue
            shift = key[position % keylength]
            c = (p + shift) % len(self.alphabet)
            ciphertext.append(c)
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        keylength = len(key)
        plaintext = []
        for position in range(0, len(ciphertext)):
            c = ciphertext[position]
            if (c > len(self.alphabet)):
                plaintext.append(self.unknown_symbol_number)
                continue
            shift = key[position % keylength]
            p = (c - shift) % len(self.alphabet)
            while p < 0:
                p = p + len(self.alphabet)
            plaintext.append(p)
        return np.array(plaintext)