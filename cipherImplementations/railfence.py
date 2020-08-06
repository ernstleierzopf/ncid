import numpy as np
from cipherImplementations.cipher import Cipher
import random
from collections import deque


class Railfence(Cipher):
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
        return np.array([length, random.randint(0, 15)])

    def encrypt(self, plaintext, key):
        ciphertext = []
        rows = [list() for i in range(key[0])]
        pos = 0
        direction = 1
        for i in range(len(plaintext) + key[1]):
            if i >= key[1]:
                rows[pos].append(plaintext[i-key[1]])
            pos += 1 * direction
            if pos == key[0] - 1 or pos == 0:
                direction = direction * -1
        for row in rows:
            ciphertext += row
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        row_sizes = [0 for i in range(key[0])]
        pos = 0
        direction = 1
        for i in range(len(ciphertext) + key[1]):
            if i >= key[1]:
                row_sizes[pos] += 1
            pos += 1 * direction
            if pos == key[0] - 1 or pos == 0:
                direction = direction * -1
        start = 0
        rows = []
        for row_size in row_sizes:
            rows.append(deque(ciphertext[start:start+row_size]))
            start += row_size
        pos = 0
        direction = 1
        for i in range(len(ciphertext) + key[1]):
            if i >= key[1]:
                plaintext.append(rows[pos].popleft())
            pos += 1 * direction
            if pos == key[0] - 1 or pos == 0:
                direction = direction * -1
        return np.array(plaintext)