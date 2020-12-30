from cipherImplementations.cipher import Cipher, generate_random_keyword, generate_keyword_alphabet
from cipherImplementations.polybius import Polybius
from cipherImplementations.polybius_square import PolybiusSquare
import random
import numpy as np


class Checkerboard(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        if length is None or length <= 0 or length > len(self.alphabet):
            raise ValueError('The length of a key must be greater than 0 and smaller or equal the size of the alphabet.')
        if length % 5 != 0:
            raise ValueError('The length of a key must be divisible by 5.')
        rowkey = generate_random_keyword(self.alphabet, length, unique=True)
        columnkey = generate_random_keyword(self.alphabet, length, unique=True)
        alphabet = generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length))
        return [rowkey, columnkey, alphabet]

    def encrypt(self, plaintext, key):
        # the keyword column is chosen randomly. For further information see http://members.aon.at/cipherclerk/Doc/Checkerboard.html.
        __polybius = Polybius(self.alphabet, self.unknown_symbol, self.unknown_symbol_number)
        code = __polybius.encrypt(plaintext, key[2])
        ciphertext = []
        for i in range(0, len(code) - 1, 2):
            row_size = int(len(key[0]) / 5 - 1)
            column_size = int(len(key[1]) / 5 - 1)
            row = int(bytes(code[i], encoding='utf-8')) - 1
            column = int(bytes(code[i+1], encoding='utf-8')) - 1
            if row < len(key[0]) % 5:
                row_size += 1
            if column < len(key[1]) % 5:
                column_size += 1
            row += random.randint(0, row_size) * 5
            column += random.randint(0, column_size) * 5
            ciphertext.append(key[0][row])
            ciphertext.append(key[1][column])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        __polybius = Polybius(self.alphabet, self.unknown_symbol, self.unknown_symbol_number)
        square = PolybiusSquare(self.alphabet, key[2])
        plaintext = []
        for i in range(0, len(ciphertext) - 1, 2):
            row = np.where(key[0] == ciphertext[i])[0][0]
            column = np.where(key[1] == ciphertext[i+1])[0][0]
            plaintext.append(square.get_char(row, column))
        return np.array(plaintext)

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower().replace(b'j', b'i')
        plaintext = super().filter(bytes(plaintext), keep_unknown_symbols)
        return plaintext
