import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword, generate_keyword_alphabet


class Headlines(Cipher):
    """This implementation differs from the ACA-described implementation. It does not shift the setting after every plaintext, but it
    divides the plaintext in 5 equally long parts and encrypts every part with the corresponding setting."""
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        setting = generate_random_keyword(self.alphabet, 5, unique=True)
        key = generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length), indexed_kw_transposition=True,
                                        second_index_kw=generate_random_keyword(self.alphabet, length))
        return [setting, key]

    def encrypt(self, plaintext, key):
        if len(plaintext) % 5 != 0:
            raise ValueError('The length of the plaintext must be divisible by 5.')
        split_size = len(plaintext) / 5
        setting = key[0]
        key = key[1]
        ciphertext = []
        for i, p in enumerate(plaintext):
            ciphertext.append(key[(np.where(key == p)[0][0] + np.where(key == setting[int(i / split_size)])[0][0]) % len(key)])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        if len(ciphertext) % 5 != 0:
            raise ValueError('The length of the ciphertext must be divisible by 5.')
        split_size = len(ciphertext) / 5
        setting = key[0]
        key = key[1]
        plaintext = []
        for i, c in enumerate(ciphertext):
            plaintext.append(key[(np.where(key == c)[0][0] - np.where(key == setting[int(i / split_size)])[0][0]) % len(key)])
        return np.array(plaintext)