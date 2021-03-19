from cipherImplementations.cipher import Cipher, generate_random_keyword
import numpy as np
from util.utils import map_text_into_numberspace


class Null(Cipher):
    """This implementation takes off ciphertext by columns."""

    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        return None

    def encrypt(self, plaintext, key):
        ciphertext = []
        for p in plaintext:
            word = generate_random_keyword(self.alphabet.replace(b' ', b''), None, middle_char=self.alphabet[p]) + b' '
            ciphertext += list(map_text_into_numberspace(word, self.alphabet, self.unknown_symbol_number))
        ciphertext[:-1] = ciphertext[:-1][:-1]
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        ct = [] + list(ciphertext)
        plaintext = []
        cntr = 0
        for c in ciphertext:
            cntr += 1
            if c == 26:
                plaintext.append(ct[int(cntr / 2) + (cntr % 2 > 0) - 1])
                ct = ct[cntr:]
                cntr = 0
        plaintext.append(ct[int(cntr / 2) + (cntr % 2 > 0) - 1])
        return np.array(plaintext)
