import random
from cipherImplementations.cipher import Cipher, generate_random_keyword, OUTPUT_ALPHABET
import numpy as np


class Grandpre(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        alphabet2 = b'' + self.alphabet * 50
        key = self.alphabet
        for _ in range(56 - len(self.alphabet)):
            position = int(random.randrange(0, len(alphabet2)))
            char = bytes([alphabet2[position]])
            key = key + char
            alphabet2 = alphabet2[0:position:] + alphabet2[position+1::]
        key = list(key)
        random.shuffle(key)
        new_key = generate_random_keyword(self.alphabet, 8)
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
            ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(key[p][rand][0]), 'utf-8')))
            ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(key[p][rand][1]), 'utf-8')))
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        values = list(key.values())
        for i in range(0, len(ciphertext), 2):
            row = int(bytes([OUTPUT_ALPHABET[ciphertext[i]]]))
            column = int(bytes([OUTPUT_ALPHABET[ciphertext[i+1]]]))
            j = 0
            for j, val in enumerate(values):
                if (row, column) in val:
                    break
            plaintext.append(j)
        return np.array(plaintext)
