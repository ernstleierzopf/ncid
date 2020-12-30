import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword


class Vigenere(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        return generate_random_keyword(self.alphabet, length)

    def encrypt(self, plaintext, key):
        key_length = len(key)
        ciphertext = []
        for position in range(len(plaintext)):
            p = plaintext[position]
            if p >= len(self.alphabet):
                ciphertext.append(self.unknown_symbol_number)
                continue
            shift = key[(position - ciphertext.count(self.unknown_symbol_number)) % key_length]
            c = (p + shift) % len(self.alphabet)
            ciphertext.append(c)
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        key_length = len(key)
        plaintext = []
        for position in range(len(ciphertext)):
            c = ciphertext[position]
            if c > len(self.alphabet):
                plaintext.append(self.unknown_symbol_number)
                continue
            shift = key[(position - plaintext.count(self.unknown_symbol_number)) % key_length]
            p = (c - shift) % len(self.alphabet)
            while p < 0:
                p = p + len(self.alphabet)
            plaintext.append(p)
        return np.array(plaintext)
