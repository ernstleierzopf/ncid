import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword


class KeyPhrase(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        return generate_random_keyword(self.alphabet.replace(b' ', b''), len(self.alphabet) - 1)

    def encrypt(self, plaintext, key):
        ciphertext = []
        for position in range(0, len(plaintext)):
            p = plaintext[position]
            if p == self.alphabet.index(b' '):
                ciphertext.append(p)
                continue
            ciphertext.append(key[p])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        raise Exception("Decryption of the Key Phrase cipher is not possible.")
