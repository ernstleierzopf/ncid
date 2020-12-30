import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_list_of_unique_digits


class Myszkowski(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        if length is None or length <= 0:
            raise ValueError('The length of a key must be greater than 0 and must not be None.')
        if not isinstance(length, int):
            raise ValueError('Length must be of type integer.')
        # if the length is not even add 1.
        addition = length % 2
        value_size = int(length / 2) + addition
        key = list(generate_random_list_of_unique_digits(value_size)) + list(generate_random_list_of_unique_digits(value_size))
        if addition == 1:
            key = key[:-1]
        return np.array(key)

    def encrypt(self, plaintext, key):
        ciphertext = []
        for i in range(max(key) + 1):
            if i not in key:
                continue
            for p in range(len(plaintext)):
                if key[p % len(key)] == i:
                    ciphertext.append(plaintext[p])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = [b'']*len(ciphertext)
        pos = 0
        for i in range(max(key) + 1):
            if i not in key:
                continue
            for c in range(len(ciphertext)):
                if key[c % len(key)] == i:
                    plaintext[c] = ciphertext[pos]
                    pos += 1
        return np.array(plaintext)
