from cipherImplementations.cipher import Cipher
from cipherImplementations.polybius_square import PolybiusSquare


class Polybius(Cipher):
    """Adapted implementation from https://github.com/tigertv/secretpy"""

    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        raise Exception('This method is not allowed in this class!')

    def __enc_dec(self, alphabet, text, key, is_encrypt=True):
        square = PolybiusSquare(alphabet, key)
        res = ''
        header = range(1, square.get_columns() + 1)
        header = "".join(map(str, header))
        if is_encrypt:
            for char in text:
                coords = square.get_coordinates(char)
                row = coords[0]
                column = coords[1]
                res += header[row] + header[column]
        else:
            res = []
            for i in range(0, len(text), 2):
                try:
                    row = header.index(text[i])
                except ValueError:
                    wrchar = text[i].encode('utf-8')
                    raise Exception("Can't find char '" + wrchar + "' of text in alphabet!")
                try:
                    column = header.index(text[i + 1])
                except ValueError:
                    wrchar = text[i+1].encode('utf-8')
                    raise Exception("Can't find char '" + wrchar + "' of text in alphabet!")
                res.append(square.get_char(row, column))
        return res

    def encrypt(self, plaintext, key=None):
        return self.__enc_dec(self.alphabet, plaintext, key, True)

    def decrypt(self, ciphertext, key=None):
        return self.__enc_dec(self.alphabet, ciphertext, key, False)
