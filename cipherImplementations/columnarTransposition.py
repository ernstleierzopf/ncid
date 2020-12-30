import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_list_of_unique_digits


class ColumnarTransposition(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number, fill_blocks):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number
        self.fill_blocks = fill_blocks

    def generate_random_key(self, length):
        return generate_random_list_of_unique_digits(length)

    def encrypt(self, plaintext, key):
        ciphertext = []
        if self.fill_blocks:
            while len(plaintext) % len(key) != 0:
                if not isinstance(plaintext, list):
                    plaintext = list(plaintext)
                plaintext.append(self.alphabet.index(b'x'))
        for start in range(len(key)):
            position = np.where(key == start)[0][0]
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
        for start in range(len(key)):
            position = np.where(key == start)[0][0]
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
