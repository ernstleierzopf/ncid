import random
import numpy as np
from cipherImplementations.cipher import Cipher
import sys

sys.path.append("../../../")
from utilities import text_utils

class Monoalphabetic_substitution(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        alphabet2 = b'' + self.alphabet
        key = b''
        for i in range(len(self.alphabet)):
            position = int(random.randrange(0, len(alphabet2)))
            char = bytes([alphabet2[position]])
            key = key + char
            alphabet2 = alphabet2.replace(char, b'')
        return key

    def encrypt(self, plaintext, key):
        ciphertext = []
        for position in range(0, len(plaintext)):
            p = plaintext[position]
            if (p > len(self.alphabet)):
                ciphertext.append(self.unknown_symbol_number)
                continue
            c = np.where(key == p)[0][0]
            ciphertext.append(c)
        return np.array(ciphertext)

    def filter(self, plaintext):
        return text_utils.remove_unknown_symbols(plaintext, self.alphabet)