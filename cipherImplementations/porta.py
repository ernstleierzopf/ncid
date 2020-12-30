import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword


class Porta(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        return generate_random_keyword(self.alphabet, length)

    def encrypt(self, plaintext, key):
        return enc_dec(plaintext, key)

    def decrypt(self, ciphertext, key):
        return enc_dec(ciphertext, key)


def enc_dec(text, key):
    text2 = []
    for i, p in enumerate(text):
        if p < 13:
            val = p + int(key[i % len(key)] / 2)
            if val < 13:
                val += 13
        else:
            val = p - int(key[i % len(key)] / 2)
            if val >= 13:
                val -= 13
        text2.append(val)
    return np.array(text2)
