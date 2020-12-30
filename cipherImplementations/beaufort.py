from cipherImplementations.cipher import Cipher, generate_random_keyword
import numpy as np


class Beaufort(Cipher):
    """Adapted implementation from https://github.com/tigertv/secretpy"""

    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        return generate_random_keyword(self.alphabet, length)

    def encrypt(self, plaintext, key):
        return self.__enc_dec(self.alphabet, plaintext, key)

    def decrypt(self, ciphertext, key):
        return self.__enc_dec(self.alphabet, ciphertext, key)

    def __enc_dec(self, alphabet, text, key):
        ans = []
        for i in range(len(text)):
            char = text[i]
            keychar = key[i % len(key)]
            ans.append((keychar - char) % len(alphabet))
        return np.array(ans)
