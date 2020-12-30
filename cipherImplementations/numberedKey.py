import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword, OUTPUT_ALPHABET
import random


class NumberedKey(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        kw = generate_random_keyword(self.alphabet, length)
        alphabet2 = b'' + self.alphabet
        key = kw
        for c in alphabet2:
            c = bytes([c])
            if c not in key:
                key += c
        rand = random.randint(1, len(key) - 1)
        shifted_key = b''
        for i in range(len(key)):
            c = bytes([key[(i + rand) % len(key)]])
            shifted_key += c
        return shifted_key

    def encrypt(self, plaintext, key):
        ciphertext = []
        for p in plaintext:
            pos = np.where(key == p)[0]
            rand = random.randint(0, len(pos) - 1)
            pos = pos[rand]
            ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(int(pos / 10)), 'utf-8')))
            ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(pos % 10), 'utf-8')))
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        ciphertext = np.array([int(bytes([OUTPUT_ALPHABET[c]])) for c in ciphertext])
        plaintext = []
        for i in range(0, len(ciphertext), 2):
            pos = ciphertext[i] * 10 + ciphertext[i + 1]
            plaintext.append(key[pos])
        return np.array(plaintext)
