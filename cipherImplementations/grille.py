from cipherImplementations.cipher import Cipher
import random
import numpy as np


class Grille(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        if length is None or length <= 1:
            raise ValueError('The length of a key must be greater than 1 and must not be None.')
        if not isinstance(length, int):
            raise ValueError('Length must be of type integer.')
        key = [[0]*length for _ in range(length)]
        rand_values = 0
        while rand_values < length:
            r = random.randint(0, (length * length) - 1)
            if key[int(r / length)][r % length] == 0:
                key[int(r / length)][r % length] = 1
                rand_values += 1
        return np.array(key, int)

    def encrypt(self, plaintext, key):
        length = len(key)
        square_size = length * length
        if len(plaintext) % square_size != 0:
            AttributeError('The Grille cipher needs the plaintext to be divisible by the squared key. Plaintext size: %d, square size:'
                           ' %dx%d' % (len(plaintext), len(key), len(key)))

        ciphertext = []
        for i in range(int(len(plaintext) / square_size)):
            ct = [[0] * length for _ in range(length)]
            count = 0
            for _ in range(length):
                for j in range(square_size):
                    if key[int(j / length)][j % length] == 1:
                        shift = 0
                        if i > 0:
                            shift = 1
                        ct[int(j / length)][j % length] = plaintext[i * square_size + count - shift]
                        count += 1
                        if count % length == 0:
                            break
                # 3 times rotated as numpy's rot90 function goes counter clock wise, but we need clockwise rotation
                key = np.rot90(key, 3)
            ciphertext += list(np.array(ct).flatten())
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        length = len(key)
        square_size = length * length
        if len(ciphertext) % square_size != 0:
            AttributeError('The Grille cipher needs the ciphertext to be divisible by the squared key. Plaintext size: %d, square size:'
                           ' %dx%d' % (len(ciphertext), len(key), len(key)))

        plaintext = []
        for i in range(int(len(ciphertext) / square_size)):
            count = 0
            for _ in range(length):
                for j in range(square_size):
                    if key[int(j / length)][j % length] == 1:
                        plaintext.append(ciphertext[i * square_size + j])
                        count += 1
                        if count % length == 0:
                            break
                # 3 times rotated as numpy's rot90 function goes counter clock wise, but we need clockwise rotation
                key = np.rot90(key, 3)
        return np.array(plaintext).flatten()
