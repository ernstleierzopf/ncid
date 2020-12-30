import numpy as np
from cipherImplementations.cipher import Cipher


class RouteTransposition(Cipher):
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
        return length

    def encrypt(self, plaintext, key):
        if len(plaintext) % key != 0:
            raise ValueError('The length of the plaintext must be divisible by the key. In this case the plaintext length is %d and the '
                             'key is %d' % (len(plaintext), key))
        ciphertext = []
        matrix = [list(range(i, i + key, 1)) for i in range(0, len(plaintext), key)]
        positions = []
        row = int(len(plaintext) / key)
        column = key
        for line in range(1, (row + column)):
            start_col = max(0, line - row)
            count = min(line, (column - start_col), row)

            lst = []
            for j in range(0, count):
                lst.append(matrix[min(row, line) - j - 1][start_col + j])
            positions.append(lst)

        # read out in spiral form
        for i in range(1, len(positions), 2):
            positions[i].reverse()
        matrix = [[0]*key for _ in range(int(len(plaintext) / 3))]
        cntr = 0
        for pos in positions:
            for i in pos:
                matrix[int(i / key)][i % key] = plaintext[cntr]
                cntr += 1

        k = 0
        l1 = 0
        m = len(matrix)
        n = key
        a = matrix

        while k < m and l1 < n:
            for i in range(l1, n):
                ciphertext.append(a[k][i])
            k += 1

            for i in range(k, m):
                ciphertext.append(a[i][n - 1])
            n -= 1

            if k < m:
                for i in range(n - 1, (l1 - 1), -1):
                    ciphertext.append(a[m - 1][i])
                m -= 1

            if l1 < n:
                for i in range(m - 1, k - 1, -1):
                    ciphertext.append(a[i][l1])
                l1 += 1
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        raise Exception('not implemented yet...')
