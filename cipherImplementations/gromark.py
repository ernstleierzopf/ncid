import random
import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword, generate_keyword_alphabet, OUTPUT_ALPHABET


class Gromark(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        primer = []
        for _ in range(5):
            primer.append(random.randint(0, 9))
        key = generate_keyword_alphabet(
            self.alphabet, generate_random_keyword(self.alphabet, length, unique=True), indexed_kw_transposition=True)
        return [primer, key]

    def encrypt(self, plaintext, key):
        primer = key[0]
        i = 0
        while len(primer) < len(plaintext):
            primer.append((primer[i] + primer[i+1]) % 10)
            i += 1

        ciphertext = []
        for k in primer[:5]:
            ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(k), 'utf-8')))
        for i, p in enumerate(plaintext):
            ciphertext.append(key[1][(p + primer[i]) % len(self.alphabet)])
        ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(primer[i]), 'utf-8')))
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        primer = key[0]
        i = 0
        while len(primer) < len(ciphertext):
            primer.append((primer[i] + primer[i + 1]) % 10)
            i += 1

        plaintext = []
        for i, c in enumerate(ciphertext[5:-1]):
            plaintext.append((np.where(key[1] == c)[0][0] - primer[i]) % len(self.alphabet))
        return np.array(plaintext)
