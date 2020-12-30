import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_list_of_unique_digits
import random
from collections import deque


class Redefence(Cipher):
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
        return [(generate_random_list_of_unique_digits(length)), random.randint(0, 15)]

    def encrypt(self, plaintext, key):
        ciphertext = []
        row_size = len(key[0])
        rows = [[] for _ in range(row_size)]
        pos = 0
        direction = 1
        for i in range(len(plaintext) + key[1]):
            if i >= key[1]:
                rows[pos].append(plaintext[i-key[1]])
            pos += 1 * direction
            if pos in (row_size - 1, 0):
                direction = direction * -1
        for i in range(len(rows)):
            ciphertext += rows[np.where(key[0] == i)[0][0]]
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        size = len(key[0])
        row_sizes = [0 for _ in range(size)]
        pos = 0
        direction = 1
        for i in range(len(ciphertext) + key[1]):
            if i >= key[1]:
                row_sizes[pos] += 1
            pos += 1 * direction
            if pos in (size - 1, 0):
                direction = direction * -1
        start = 0
        row_sizes = [row_sizes[np.where(key[0] == i)[0][0]] for i in range(len(key[0]))]
        rows = []
        for row_size in row_sizes:
            rows.append(deque(ciphertext[start:start+row_size]))
            start += row_size
        rows = [rows[np.where(key[0] == i)[0][0]] for i in range(len(key[0]))]
        pos = 0
        direction = 1
        for i in range(len(ciphertext) + key[1]):
            if i >= key[1]:
                plaintext.append(rows[pos].popleft())
            pos += 1 * direction
            if pos in (size - 1, 0):
                direction = direction * -1
        return np.array(plaintext)
