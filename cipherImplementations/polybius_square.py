from util.textUtils import map_text_into_numberspace
import math
from collections import OrderedDict


class PolybiusSquare:
    """PolybiusSquare. It's used by many classical ciphers"""
    __alphabet = None
    __side = 0

    def __init__(self, alphabet, key=None):
        keyi = []
        if key is not None:
            for char in key:
                index = self.__find_index_in_alphabet(char, alphabet)
                keyi.append(index)
            # remove duplicates
            keyi = OrderedDict.fromkeys(keyi)

        alph_out = bytearray()
        for i in keyi:
            alph_out.append(alphabet[i])

        for i in range(len(alphabet)):
            if i not in keyi:
                alph_out.append(alphabet[i])

        self.__alphabet = map_text_into_numberspace(bytes(alph_out), alphabet, 90)
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