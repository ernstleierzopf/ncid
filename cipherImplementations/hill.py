import numpy as np
from cipherImplementations.cipher import Cipher
import random


class Hill(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        matrix = np.zeros((4, 4), dtype=int)
        for x in range(0, 4):
            for y in range(0, 4):
                matrix[x, y] = int(random.randrange(0, 25))
        det = self.determinant(matrix)
        while det == 0 or self.euclid_algo(det, 26) > 1:
            x = random.randrange(0, 4)
            y = random.randrange(0, 4)
            matrix[x, y] = int(random.randrange(0, 25))
            det = self.determinant(matrix)
        return np.array(matrix)

    def encrypt(self, plaintext, key):
        ciphertext = []
        for position in range(3, len(plaintext), 4):
            p = [plaintext[position-3], plaintext[position-2], plaintext[position-1], plaintext[position]]
            c = np.matmul(key, p)
            c = c % 26
            for n in c:
                ciphertext.append(n)
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        raise Exception('Not implmplemented yet..')

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = super().filter(plaintext, keep_unknown_symbols)
        if keep_unknown_symbols:
            i = 0
            while i < len(plaintext):
                if plaintext[i] not in self.alphabet:
                    plaintext = plaintext.replace(bytes([plaintext[i]]), b'x')
                else:
                    i += 1

        # while len(plaintext) % 4 != 0:
        #    plaintext += b'x'
        return plaintext

    def determinant(self, matrix):
        return int((
            matrix[0, 3] * matrix[1, 2] * matrix[2, 1] * matrix[3, 0] - matrix[0, 2] * matrix[1, 3] * matrix[2, 1] * matrix[3, 0] -
            matrix[0, 3] * matrix[1, 1] * matrix[2, 2] * matrix[3, 0] + matrix[0, 1] * matrix[1, 3] * matrix[2, 2] * matrix[3, 0] +
            matrix[0, 2] * matrix[1, 1] * matrix[2, 3] * matrix[3, 0] - matrix[0, 1] * matrix[1, 2] * matrix[2, 3] * matrix[3, 0] -
            matrix[0, 3] * matrix[1, 2] * matrix[2, 0] * matrix[3, 1] + matrix[0, 2] * matrix[1, 3] * matrix[2, 0] * matrix[3, 1] +
            matrix[0, 3] * matrix[1, 0] * matrix[2, 2] * matrix[3, 1] - matrix[0, 0] * matrix[1, 3] * matrix[2, 2] * matrix[3, 1] -
            matrix[0, 2] * matrix[1, 0] * matrix[2, 3] * matrix[3, 1] + matrix[0, 0] * matrix[1, 2] * matrix[2, 3] * matrix[3, 1] +
            matrix[0, 3] * matrix[1, 1] * matrix[2, 0] * matrix[3, 2] - matrix[0, 1] * matrix[1, 3] * matrix[2, 0] * matrix[3, 2] -
            matrix[0, 3] * matrix[1, 0] * matrix[2, 1] * matrix[3, 2] + matrix[0, 0] * matrix[1, 3] * matrix[2, 1] * matrix[3, 2] +
            matrix[0, 1] * matrix[1, 0] * matrix[2, 3] * matrix[3, 2] - matrix[0, 0] * matrix[1, 1] * matrix[2, 3] * matrix[3, 2] -
            matrix[0, 2] * matrix[1, 1] * matrix[2, 0] * matrix[3, 3] + matrix[0, 1] * matrix[1, 2] * matrix[2, 0] * matrix[3, 3] +
            matrix[0, 2] * matrix[1, 0] * matrix[2, 1] * matrix[3, 3] - matrix[0, 0] * matrix[1, 2] * matrix[2, 1] * matrix[3, 3] -
            matrix[0, 1] * matrix[1, 0] * matrix[2, 2] * matrix[3, 3] + matrix[0, 0] * matrix[1, 1] * matrix[2, 2] * matrix[3, 3]) %
                   len(self.alphabet))

    # compute greatest common divisor using euclid algorithm
    def euclid_algo(self, x, y):
        if x < y:  # We want x >= y
            return self.euclid_algo(y, x)
        while y != 0:
            (x, y) = (y, x % y)
        return x
