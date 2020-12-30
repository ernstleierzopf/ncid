import math


class PolybiusSquare:
    """PolybiusSquare. It's used by many classical ciphers"""

    __alphabet = None
    __side = 0

    def __init__(self, alphabet, key=None):
        self.__alphabet = key
        self.__side = int(math.ceil(math.sqrt(len(alphabet))))

    def __find_index_in_alphabet(self, char, alphabet):
        for j in range(len(alphabet)):
            if alphabet[j] == char:
                break
        return j

    def get_coordinates(self, char):
        for j in range(len(self.__alphabet)):
            if self.__alphabet[j] == char:
                break
        row = int(j / self.__side)
        column = j % self.__side
        return row, column

    def get_char(self, row, column):
        return self.__alphabet[row * self.__side + column]

    def get_columns(self):
        return self.__side

    def get_rows(self):
        return int(len(self.__alphabet) / self.__side)
