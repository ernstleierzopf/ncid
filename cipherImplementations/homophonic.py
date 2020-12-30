import random
import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword, OUTPUT_ALPHABET


class Homophonic(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        return generate_random_keyword(self.alphabet, 4, unique=True)

    def encrypt(self, plaintext, key):
        ciphertext = []
        for p in plaintext:
            rand = random.randint(0, 3)
            if p >= key[rand]:
                ct = p - key[rand] + 25 * rand + 1
            else:
                ct = len(self.alphabet) + p - key[rand] + 25 * rand + 1
            ct = ct % 100
            ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(int(ct / 10)), 'utf-8')))
            ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(ct % 10), 'utf-8')))
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        for i in range(0, len(ciphertext), 2):
            ct = int(bytes([OUTPUT_ALPHABET[ciphertext[i]]])) * 10 + int(bytes([OUTPUT_ALPHABET[ciphertext[i + 1]]])) - 1
            if ct == -1:
                ct = 99
            rand = int(ct / 25)
            if ct < key[rand]:
                p = ct + key[rand] - 25 * rand
            else:
                p = (ct - len(self.alphabet) + key[rand] + 25 * rand) % len(self.alphabet)
            plaintext.append(p)
        return np.array(plaintext)

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower().replace(b'j', b'i')
        plaintext = super().filter(bytes(plaintext), keep_unknown_symbols)
        return plaintext
