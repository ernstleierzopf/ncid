import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword, generate_keyword_alphabet


def get_right_neighbor(index):
    if index % 5 < 4:
        return index + 1
    if index % 5 == 4:
        return index - 4
    return -1


def get_lower_neighbour(index):
    return (index + 5) % 25


def get_substitute(row, col):
    return 5 * row + col


def get_left_neighbor(index):
    if index % 5 > 0:
        return index - 1
    if index % 5 == 0:
        return index + 4
    return -1


def get_upper_neighbour(index):
    return (index - 5) % 25


class Playfair(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        if length is None or length >= len(self.alphabet):
            raise ValueError('The length must not be greater than the length of the alphabet.')
        return generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length))

    def encrypt(self, plaintext, key):
        ciphertext = []
        for position in range(1, len(plaintext), 2):
            p0, p1 = plaintext[position-1], plaintext[position]
            index0 = np.where(key == p0)[0][0]
            index1 = np.where(key == p1)[0][0]
            row_p0 = int(index0 / 5)
            row_p1 = int(index1 / 5)
            col_p0 = index0 % 5
            col_p1 = index1 % 5
            if row_p0 == row_p1:
                index0 = get_right_neighbor(index0)
                index1 = get_right_neighbor(index1)
            elif col_p0 == col_p1:
                index0 = get_lower_neighbour(index0)
                index1 = get_lower_neighbour(index1)
            else:
                index0 = get_substitute(row_p0, col_p1)
                index1 = get_substitute(row_p1, col_p0)
            ciphertext.append(key[int(index0)])
            ciphertext.append(key[int(index1)])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        for position in range(1, len(ciphertext), 2):
            c0, c1 = ciphertext[position - 1], ciphertext[position]
            index0 = np.where(key == c0)[0][0]
            index1 = np.where(key == c1)[0][0]
            row_p0 = int(index0 / 5)
            row_p1 = int(index1 / 5)
            col_p0 = index0 % 5
            col_p1 = index1 % 5
            if row_p0 == row_p1:
                index0 = get_left_neighbor(index0)
                index1 = get_left_neighbor(index1)
            elif col_p0 == col_p1:
                index0 = get_upper_neighbour(index0)
                index1 = get_upper_neighbour(index1)
            else:
                index0 = get_substitute(row_p0, col_p1)
                index1 = get_substitute(row_p1, col_p0)
            plaintext.append(key[int(index0)])
            plaintext.append(key[int(index1)])
        return np.array(plaintext)

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower().replace(b'j', b'i')
        plaintext = super().filter(bytes(plaintext), keep_unknown_symbols)
        if len(plaintext) == 0:
            return b''
        output = bytearray()
        for position in range(1, len(plaintext)):
            p0, p1 = plaintext[position - 1], plaintext[position]
            output.append(p0)
            if p0 == p1:
                output.append(120)  # 120 = 'x'
        output.append(plaintext[len(plaintext)-1])
        plaintext = bytes(output)
        return plaintext
