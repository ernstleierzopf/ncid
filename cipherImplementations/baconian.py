from cipherImplementations.cipher import Cipher
import random


class Baconian(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number, split_plaintext=True):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number
        self.split_plaintext = split_plaintext

    def generate_random_key(self, length=None):
        return [[0,0,0,0,0], [0,0,0,0,1], [0,0,0,1,0], [0,0,0,1,1], [0,0,1,0,0], [0,0,1,0,1], [0,0,1,1,0], [0,0,1,1,1], [0,1,0,0,0],
                [0,1,0,0,0], [0,1,0,0,1], [0,1,0,1,0], [0,1,0,1,1], [0,1,1,0,0], [0,1,1,0,1], [0,1,1,1,0], [0,1,1,1,1], [1,0,0,0,0],
                [1,0,0,0,1], [1,0,0,1,0], [1,0,0,1,1], [1,0,0,1,1], [1,0,1,0,0], [1,0,1,0,1], [1,0,1,1,0], [1,0,1,1,1]]

    def encrypt(self, plaintext, key):
        if self.split_plaintext:
            plaintext = plaintext[:int(len(plaintext) / 5)]
        ciphertext = []
        for p in plaintext:
            for k in key[p]:
                if k < 13:
                    ciphertext.append(random.randint(0, 12))
                else:
                    ciphertext.append(random.randint(13, 25))
        return ciphertext

    def decrypt(self, ciphertext, key):
        plaintext = []
        tmp = []
        for c in ciphertext:
            tmp.append(int(c / 13))
            if len(tmp) == 5:
                plaintext.append(key.index(tmp))
                tmp = []
        return plaintext