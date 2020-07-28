from cipherImplementations.cipher import Cipher
from cipherImplementations.polybius_square import PolybiusSquare
import random


class Foursquare(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        alphabet2 = b'' + self.alphabet
        key1 = b''
        for _ in range(len(self.alphabet)):
            position = int(random.randrange(0, len(alphabet2)))
            char = bytes([alphabet2[position]])
            key1 = key1 + char
            alphabet2 = alphabet2.replace(char, b'')

        alphabet2 = b'' + self.alphabet
        key2 = b''
        for _ in range(len(self.alphabet)):
            position = int(random.randrange(0, len(alphabet2)))
            char = bytes([alphabet2[position]])
            key2 = key2 + char
            alphabet2 = alphabet2.replace(char, b'')
        return key1, key2

    def encrypt(self, plaintext, key):
        square01 = PolybiusSquare(self.alphabet, key[0])
        square10 = PolybiusSquare(self.alphabet, key[1])
        square = PolybiusSquare(self.alphabet, None)

        odd = plaintext[1::2]
        even = plaintext[::2]
        ciphertext = []

        for i in range(len(even)):
            coords = square.get_coordinates(even[i])
            row00 = coords[0]
            column00 = coords[1]

            coords = square.get_coordinates(odd[i])
            row11 = coords[0]
            column11 = coords[1]

            ciphertext.append(square01.get_char(row00, column11))
            ciphertext.append(square10.get_char(row11, column00))
        return ciphertext

    def decrypt(self, ciphertext, key):
        square01 = PolybiusSquare(self.alphabet, key[0])
        square10 = PolybiusSquare(self.alphabet, key[1])
        square = PolybiusSquare(self.alphabet, None)

        odd = ciphertext[1::2]
        even = ciphertext[::2]
        plaintext = []
        for i in range(len(even)):
            coords = square01.get_coordinates(even[i])
            row00 = coords[0]
            column00 = coords[1]

            coords = square10.get_coordinates(odd[i])
            row11 = coords[0]
            column11 = coords[1]

            plaintext.append(square.get_char(row00, column11))
            plaintext.append(square.get_char(row11, column00))
        return plaintext

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower().replace(b'j', b'i')
        plaintext = super().filter(bytes(plaintext), keep_unknown_symbols)
        return plaintext