import random
import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword, generate_keyword_alphabet, OUTPUT_ALPHABET
from util.utils import map_text_into_numberspace


class PeriodicGromark(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        if length is None or length <= 0:
            raise ValueError('The length of a key must be greater than 0 and must not be None.')
        if not isinstance(length, int):
            raise ValueError('Length must be of type integer.')
        periodic_key = []
        for _ in range(length):
            periodic_key.append(random.randint(0, len(self.alphabet) - 1))
        kw = generate_random_keyword(self.alphabet, length, unique=True)
        indices = np.argsort(map_text_into_numberspace(kw, self.alphabet, self.unknown_symbol_number))
        primer = [0]*len(indices)
        for number, index in enumerate(indices):
            primer[index] = number + 1
        key = generate_keyword_alphabet(self.alphabet, kw, indexed_kw_transposition=True)
        return [primer, np.array(periodic_key), key]

    def encrypt(self, plaintext, key):
        primer = list(key[0])
        periodic_key = key[1]
        ciphertext = []
        for k in primer[:len(key[0])]:
            ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(k), 'utf-8')))

        i = 0
        while len(primer) < len(plaintext):
            primer.append((primer[i] + primer[i+1]) % 10)
            i += 1

        for i, p in enumerate(plaintext):
            ciphertext.append(key[2][(p + primer[i] + periodic_key[int(i / len(periodic_key)) % len(periodic_key)]) % len(self.alphabet)])
        ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(primer[i]), 'utf-8')))
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        primer = list(key[0])
        periodic_key = key[1]
        length = len(primer)
        i = 0
        while len(primer) < len(ciphertext):
            primer.append((primer[i] + primer[i + 1]) % 10)
            i += 1

        plaintext = []
        for i, c in enumerate(ciphertext[length:-1]):
            plaintext.append((np.where(key[2] == c)[0][0] - primer[i] - periodic_key[int(i / len(periodic_key)) % len(
                periodic_key)]) % len(self.alphabet))
        return np.array(plaintext)
