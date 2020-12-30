import random
import numpy as np
from cipherImplementations.cipher import Cipher


class SimpleSubstitution(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        alphabet2 = b'' + self.alphabet
        key = b''
        for _ in range(len(self.alphabet)):
            position = int(random.randrange(0, len(alphabet2)))
            char = bytes([alphabet2[position]])
            key = key + char
            alphabet2 = alphabet2.replace(char, b'')
        return key

    def encrypt(self, plaintext, key):
        ciphertext = []
        for position in range(0, len(plaintext)):
            p = plaintext[position]
            if p >= len(self.alphabet):
                ciphertext.append(self.unknown_symbol_number)
                continue
            ciphertext.append(key[p])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        for position in range(0, len(ciphertext)):
            c = ciphertext[position]
            if c >= len(self.alphabet):
                plaintext.append(self.unknown_symbol_number)
                continue
            p = np.where(key == c)[0][0]
            while p < 0:
                p = p + len(self.alphabet)
            plaintext.append(p)
        return np.array(plaintext)
