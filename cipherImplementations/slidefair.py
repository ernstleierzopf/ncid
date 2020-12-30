import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword


class Slidefair(Cipher):
    """This implementation takes the ciphertext off in rows."""

    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        return generate_random_keyword(self.alphabet, length)

    def encrypt(self, plaintext, key):
        ciphertext = []
        key_pos = 0
        for i in range(0, len(plaintext), 2):
            c1 = (plaintext[i+1] - key[key_pos]) % len(self.alphabet)
            c2 = (plaintext[i] + key[key_pos]) % len(self.alphabet)
            if c1 == plaintext[i] and c2 == plaintext[i+1]:
                c1 = (c1 + 1) % len(self.alphabet)
                c2 = (c2 + 1) % len(self.alphabet)
            ciphertext.append(c1)
            ciphertext.append(c2)
            key_pos = (key_pos + 1) % len(key)
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        key_pos = 0
        for i in range(0, len(ciphertext), 2):
            p1 = (ciphertext[i + 1] - key[key_pos]) % len(self.alphabet)
            p2 = (ciphertext[i] + key[key_pos]) % len(self.alphabet)
            if p1 == ciphertext[i] and p2 == ciphertext[i + 1]:
                p1 = (p1 - 1) % len(self.alphabet)
                p2 = (p2 - 1) % len(self.alphabet)
            plaintext.append(p1)
            plaintext.append(p2)
            key_pos = (key_pos + 1) % len(key)
        return np.array(plaintext)
