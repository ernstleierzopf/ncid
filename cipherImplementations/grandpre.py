import random
from cipherImplementations.cipher import Cipher
import numpy as np


class Grandpre(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        alphabet2 = b'' + self.alphabet * 50
        key = self.alphabet
        for _ in range(64 - len(self.alphabet)):
            position = int(random.randrange(0, len(alphabet2)))
            char = bytes([alphabet2[position]])
            key = key + char
            alphabet2 = alphabet2[0:position:] + alphabet2[position+1::]
        key = list(key)
        random.shuffle(key)
        new_key = b''
        for k in key:
            new_key += bytes([k])
        key = new_key
        key_dict = {}
        for k in [bytes([i]) for i in set(key)]:
            key_dict[k] = []
        for pos, k in enumerate(key):
            row = int(pos / 8)
            column = pos % 8
            key_dict[bytes([k])].append((row, column))
        return key_dict

    def encrypt(self, plaintext, key):
        ciphertext = []
        for p in plaintext:
            rand = random.randint(0, len(key[p]) - 1)
            row = key[p][rand][0]
            column = key[p][rand][1]

            upper = 1
            if row < 6:
                upper += 1
            rand = (random.randint(0, upper))
            ciphertext.append(rand * 10 + row)

            upper = 1
            if column < 6:
                upper += 1
            rand = (random.randint(0, upper))
            ciphertext.append(rand * 10 + column)
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        values = list(key.values())
        for i in range(0, len(ciphertext), 2):
            row = ciphertext[i] % 10
            column = ciphertext[i+1] % 10
            for i, val in enumerate(values):
                if (row, column) in val:
                    break
            plaintext.append(i)
        return np.array(plaintext)