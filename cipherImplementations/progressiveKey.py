import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword
from cipherImplementations.vigenere import Vigenere


class ProgressiveKey(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number, progression_index=None):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number
        self.vigenere = Vigenere(alphabet, unknown_symbol, unknown_symbol_number)
        if progression_index is not None and progression_index <= 0:
            raise ValueError('The progression_index must be greater than zero.')
        self.progression_index = progression_index

    def generate_random_key(self, length):
        # for comfortability the progression_index is length / 2
        return generate_random_keyword(self.alphabet, length)

    def encrypt(self, plaintext, key):
        progression_index = self.progression_index
        if progression_index is None:
            progression_index = max(int(len(key) / 2), 1)
        ciphertext = self.vigenere.encrypt(plaintext, key)
        for i in range(1, progression_index + 1, 1):
            new_key = []
            for j in range(len(plaintext)):
                new_key.append((int(j / len(key)) * i) % len(self.alphabet))
            ciphertext = self.vigenere.encrypt(ciphertext, np.array(new_key))
        return ciphertext

    def decrypt(self, ciphertext, key):
        progression_index = self.progression_index
        if progression_index is None:
            progression_index = max(int(len(key) / 2), 1)
        for i in range(progression_index, 0, -1):
            new_key = []
            for j in range(len(ciphertext)):
                new_key.append((int(j / len(key)) * i) % len(self.alphabet))
            ciphertext = self.vigenere.decrypt(ciphertext, np.array(new_key))
        return self.vigenere.decrypt(ciphertext, key)
