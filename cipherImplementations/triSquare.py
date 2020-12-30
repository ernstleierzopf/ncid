from cipherImplementations.cipher import Cipher, generate_random_keyword, generate_keyword_alphabet
import random
import numpy as np


class TriSquare(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        key1 = generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length), vertical=True)
        key2 = generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length))
        key3 = generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length))
        return [key1, key2, key3]

    def encrypt(self, plaintext, key):
        if len(plaintext) % 2 != 0:
            plaintext = list(plaintext)
            plaintext.append(self.alphabet.index(b'x'))
        ciphertext = []
        for i in range(0, len(plaintext), 2):
            column1 = np.where(key[0] == plaintext[i])[0][0] % 5
            row1 = int(np.where(key[0] == plaintext[i])[0][0] / 5)
            column2 = np.where(key[1] == plaintext[i+1])[0][0] % 5
            row2 = int(np.where(key[1] == plaintext[i+1])[0][0] / 5)

            rand = random.randint(0,4)
            ciphertext.append(key[0][column1 + rand * 5])
            ciphertext.append(key[2][row1 * 5 + column2])
            rand = random.randint(0,4)
            ciphertext.append(key[1][row2 * 5 + rand])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        for i in range(0, len(ciphertext), 3):
            column1 = np.where(key[0] == ciphertext[i])[0][0] % 5
            row1 = int(np.where(key[2] == ciphertext[i+1])[0][0] / 5)
            column2 = np.where(key[2] == ciphertext[i+1])[0][0] % 5
            row2 = int(np.where(key[1] == ciphertext[i+2])[0][0] / 5)
            plaintext.append(key[0][column1 + row1 * 5])
            plaintext.append(key[1][column2 + row2 * 5])
        return np.array(plaintext)

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower().replace(b'j', b'i')
        plaintext = super().filter(bytes(plaintext), keep_unknown_symbols)
        return plaintext
