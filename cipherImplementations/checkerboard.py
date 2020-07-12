from cipherImplementations.cipher import Cipher
from cipherImplementations.polybius import Polybius
from cipherImplementations.polybius_square import PolybiusSquare
import random


class Checkerboard(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        if length is None or length <= 0 or length > len(self.alphabet):
            raise ValueError('The length of a key must be greater than 0 and smaller or equal the size of the alphabet.')
        if not isinstance(length, int):
            raise ValueError('Length must be of type integer.')
        alphabet2 = b'' + self.alphabet
        rowkey = b''
        for _ in range(length):
            position = int(random.randrange(0, len(alphabet2)))
            char = bytes([alphabet2[position]])
            rowkey = rowkey + char
            alphabet2 = alphabet2.replace(char, b'')

        alphabet2 = b'' + self.alphabet
        columnkey = b''
        for _ in range(length):
            position = int(random.randrange(0, len(alphabet2)))
            char = bytes([alphabet2[position]])
            columnkey = columnkey + char
            alphabet2 = alphabet2.replace(char, b'')

        alphabet2 = b'' + self.alphabet
        alphabet = b''
        for _ in range(len(self.alphabet)):
            position = int(random.randrange(0, len(alphabet2)))
            char = bytes([alphabet2[position]])
            alphabet = alphabet + char
            alphabet2 = alphabet2.replace(char, b'')
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
        return ciphertext

    def decrypt(self, ciphertext, key):
        __polybius = Polybius(self.alphabet, self.unknown_symbol, self.unknown_symbol_number)
        code = __polybius.encrypt(ciphertext, key[2])
        square = PolybiusSquare(self.alphabet, key[2])
        key[0] = list(key[0])
        key[1] = list(key[1])
        plaintext = []
        for i in range(0, len(ciphertext) - 1, 2):
            row = key[0].index(ciphertext[i])
            column = key[1].index(ciphertext[i+1])
            plaintext.append(square.get_char(row, column))
        return plaintext

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower().replace(b'j', b'i')
        plaintext = super().filter(bytes(plaintext), keep_unknown_symbols)
        return plaintext