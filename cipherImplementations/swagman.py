import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_list_of_unique_digits


def get_row_index(i, key):
    index = int(i / len(key)) % len(key)
    key_index = i % len(key)
    for j in range(len(key)):
        if key[j][index] == key_index:
            return j
    return 0


class Swagman(Cipher):
    """This implementation takes the ciphertext off in rows."""

    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        if length is None or length <= 0:
            raise ValueError('The length of a key must be greater than 0 and must not be None.')
        if not isinstance(length, int):
            raise ValueError('Length must be of type integer.')
        key = []
        k = []
        lst = []
        i = 0
        while i < length * length:
            while len(lst) == 0:
                lst = list(generate_random_list_of_unique_digits(length))
                for j in range(len(k)):
                    lst.remove(k[j])
                for j in range(int(i / length)):
                    char = key[j][i % length]
                    if char in lst:
                        lst.remove(char)
                if len(lst) == 0:
                    i -= len(k)
                    k = []
            k.append(lst[0])
            lst = []
            if len(k) == length:
                key.append(k)
                k = []
            i += 1
        return np.array(key)

    def encrypt(self, plaintext, key):
        plaintext = list(plaintext)
        while len(plaintext) % len(key) != 0:
            plaintext.append(self.alphabet.index(b'x'))
        plaintext = np.array(plaintext)
        distance = len(plaintext) / len(key)
        ciphertext = []
        for i in range(len(plaintext)):
            ciphertext.append(plaintext[int(i / len(key) + distance * get_row_index(i, key))])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        distance = len(ciphertext) / len(key)
        plaintext = [0]*len(ciphertext)
        for i in range(len(ciphertext)):
            plaintext[int(i / len(key) + distance * get_row_index(i, key))] = ciphertext[i]
        return np.array(plaintext)
