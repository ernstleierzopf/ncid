from cipherImplementations.cipher import Cipher, generate_random_keyword, generate_keyword_alphabet
from cipherImplementations.polybius_square import PolybiusSquare
import numpy as np


class Foursquare(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        key1 = generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length))
        key2 = generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length), vertical=True)
        return [key1, key2]

    def encrypt(self, plaintext, key):
        square01 = PolybiusSquare(self.alphabet, key[0])
        square10 = PolybiusSquare(self.alphabet, key[1])
        square = PolybiusSquare(self.alphabet, list(range(len(self.alphabet))))

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
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        square01 = PolybiusSquare(self.alphabet, key[0])
        square10 = PolybiusSquare(self.alphabet, key[1])
        square = PolybiusSquare(self.alphabet, list(range(len(self.alphabet))))

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
        return np.array(plaintext)

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower().replace(b'j', b'i')
        plaintext = super().filter(bytes(plaintext), keep_unknown_symbols)
        return plaintext
