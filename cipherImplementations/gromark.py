import random
import numpy as np
from cipherImplementations.cipher import Cipher


class Gromark(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        primer = []
        for i in range(5):
            primer.append(random.randint(0, 9))

        alphabet2 = b'' + self.alphabet
        key = b''
        for _ in range(len(self.alphabet)):
            position = int(random.randrange(0, len(alphabet2)))
            char = bytes([alphabet2[position]])
            key = key + char
            alphabet2 = alphabet2.replace(char, b'')
        return [primer, key]

    def encrypt(self, plaintext, key):
        primer = key[0]
        i = 0
        while len(primer) < len(plaintext):
            primer.append((primer[i] + primer[i+1]) % 10)
            i += 1

        ciphertext = []
        for i, p in enumerate(plaintext):
            ciphertext.append(key[1][(p + primer[i]) % len(self.alphabet)])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        primer = key[0]
        i = 0
        while len(primer) < len(ciphertext):
            primer.append((primer[i] + primer[i + 1]) % 10)
            i += 1

        plaintext = []
        for i, c in enumerate(ciphertext):
            plaintext.append((np.where(key[1] == c)[0][0] - primer[i]) % len(self.alphabet))
        return np.array(plaintext)