import random
from cipherImplementations.cipher import Cipher
import numpy as np


class MonomeDinome(Cipher):
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

        numbers = []
        for i in range(10):
            numbers.append(i)
        random.shuffle(numbers)
        return [numbers, key]

    def encrypt(self, plaintext, key):
        ciphertext = []
        for p in plaintext:
            index = np.where(key[1] == p)[0][0]
            if int(index / 8) > 0:
                ciphertext.append(key[0][int(index / 8) - 1])
            ciphertext.append(key[0][index % 8 + 2])

        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        cntr = 0
        while cntr < len(ciphertext):
            p = ciphertext[cntr]
            cntr += 1
            row = 0
            index = np.where(key[0] == p)[0][0]
            if index < 2:
                row = index + 1
                p = ciphertext[cntr]
                cntr += 1
            plaintext.append(key[1][row * 8 + np.where(key[0] == p)[0][0] - 2])
        return np.array(plaintext)

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower().replace(b'j', b'i')
        plaintext = plaintext.lower().replace(b'z', b'y')
        plaintext = super().filter(bytes(plaintext), keep_unknown_symbols)
        return plaintext