from cipherImplementations.cipher import Cipher, generate_random_list_of_unique_digits
import numpy as np


class NihilistTransposition(Cipher):
    """This implementation takes off ciphertext by columns."""

    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        return generate_random_list_of_unique_digits(length)

    def encrypt(self, plaintext, key):
        if len(plaintext) != len(key) * len(key):
            raise ValueError('The plaintext must be as long as the squared key!')
        columnar_ct = []
        for start in range(int(len(plaintext) / len(key))):
            position = start * len(key)
            for i in range(len(key)):
                columnar_ct.append(plaintext[position + np.where(key == i)[0][0]])

        ciphertext = []
        for i in range(len(key)):
            for start in range(len(key)):
                position = np.where(key == start)[0][0] * len(key) + i
                ciphertext.append(columnar_ct[position])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        if len(ciphertext) != len(key) * len(key):
            raise ValueError('The plaintext must be as long as the squared key!')
        columnar_ct = [0]*(len(key) * len(key))
        cntr = 0
        for i in range(len(key)):
            for start in range(len(key)):
                position = np.where(key == start)[0][0] * len(key) + i
                columnar_ct[position] = ciphertext[cntr]
                cntr += 1

        plaintext = [0]*(len(key) * len(key))
        cntr = 0
        for start in range(int(len(plaintext) / len(key))):
            position = start * len(key)
            for i in range(len(key)):
                plaintext[position + np.where(key == i)[0][0]] = columnar_ct[cntr]
                cntr += 1
        return np.array(plaintext)
