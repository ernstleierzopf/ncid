from cipherImplementations.cipher import Cipher, generate_keyword_alphabet
from cipherImplementations.polybius_square import PolybiusSquare
import random
import numpy as np


ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
twenties = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
thousands = ["", "thousand", "million"]


def num999(n):
    c = int(n % 10)  # singles digit
    b = int(((n % 100) - c) / 10)  # tens digit
    a = int(((n % 1000) - (b * 10) - c) / 100)  # hundreds digit
    t = ""
    h = ""
    if a != 0:
        t = ones[a] + "hundred"
    if b <= 1:
        h = ones[n % 100]
    elif b > 1:
        h = twenties[b] + ones[c]
    st = t + h
    return st


def num2word(num):
    if num == 0:
        return b'zero'
    i = 3
    n = str(num)
    word = ""
    k = 0
    while i == 3:
        nw = n[-i:]
        n = n[:-i]
        if int(nw) == 0:
            word = num999(int(nw)) + thousands[int(nw)] + word
        else:
            word = num999(int(nw)) + thousands[k] + word
        if n == '':
            i = i + 1
        k += 1
    return bytes(word, encoding='utf-8')


class Bazeries(Cipher):
    """
    Adapted implementation from https://github.com/tigertv/secretpy and https://www.quora.com/How-do-I-convert-numbers-to-words-in-Python
    """

    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        number = random.randint(1, 1000000)
        return [generate_keyword_alphabet(self.alphabet, num2word(number)), number]

    def encrypt(self, plaintext, key):
        return self.__enc_dec(self.alphabet, plaintext, key, True)

    def decrypt(self, ciphertext, key):
        return self.__enc_dec(self.alphabet, ciphertext, key, False)

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower().replace(b'j', b'i')
        plaintext = super().filter(bytes(plaintext), keep_unknown_symbols)
        return plaintext

    def __enc_dec(self, alphabet, text, key, is_encrypt=True):
        square1 = PolybiusSquare(alphabet, list(range(len(self.alphabet))))

        # key is a number, make it a string
        square2 = PolybiusSquare(alphabet, key[0])

        # prepare text: group and reverse
        temp = key[1]
        groups = []
        while temp > 0:
            rmd = temp % 10
            temp = int(temp / 10)
            groups.append(rmd)
        groups = groups[::-1]

        i = 0
        j = 0
        revtext = []
        while i < len(text):
            num = groups[j]
            str1 = list(text[int(i):int(i+num)])
            revtext += str1[::-1]
            i += num
            j += 1
            if j == len(groups):
                j = 0

        # now we have reversed text and we encrypt
        ret = []
        if is_encrypt:
            for char in revtext:
                coords = square1.get_coordinates(char)
                ret.append(square2.get_char(coords[1], coords[0]))
        else:
            for char in revtext:
                coords = square2.get_coordinates(char)
                ret.append(square1.get_char(coords[1], coords[0]))
        return np.array(ret)
