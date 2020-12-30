from cipherImplementations.cipher import Cipher
import random
import numpy as np


class Baconian(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        return [[0,0,0,0,0], [0,0,0,0,1], [0,0,0,1,0], [0,0,0,1,1], [0,0,1,0,0], [0,0,1,0,1], [0,0,1,1,0], [0,0,1,1,1], [0,1,0,0,0],
                [0,1,0,0,0], [0,1,0,0,1], [0,1,0,1,0], [0,1,0,1,1], [0,1,1,0,0], [0,1,1,0,1], [0,1,1,1,0], [0,1,1,1,1], [1,0,0,0,0],
                [1,0,0,0,1], [1,0,0,1,0], [1,0,0,1,1], [1,0,0,1,1], [1,0,1,0,0], [1,0,1,0,1], [1,0,1,1,0], [1,0,1,1,1]]

    def encrypt(self, plaintext, key):
        ciphertext = []
        for p in plaintext:
            for k in key[p]:
                if k < 13:
                    r = random.randint(0, 12)
                else:
                    r = random.randint(13, 25)
                if r in (9, 21):  # remove j and v
                    r -= 1
                ciphertext.append(r)
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        tmp = []
        for c in ciphertext:
            tmp.append(int(c / 13))
            if len(tmp) == 5:
                plaintext.append(key.index(tmp))
                tmp = []
        return np.array(plaintext)
