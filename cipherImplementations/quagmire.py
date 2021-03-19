import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword, generate_keyword_alphabet
from util.utils import map_numbers_into_textspace, map_text_into_numberspace


class Quagmire(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number, keyword_type):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number
        if not 0 < keyword_type < 5:
            raise ValueError('There is no Quagmire type %d' % keyword_type)
        self.keyword_type = keyword_type

    def generate_random_key(self, length):
        if self.keyword_type in (1, 3):
            key = [generate_random_keyword(self.alphabet, length),
                   generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length))]
        elif self.keyword_type == 2:
            key = [generate_random_keyword(self.alphabet, length),
                   generate_random_keyword(self.alphabet, length, unique=True)]
        else:
            key = [generate_random_keyword(self.alphabet, length),
                   generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length)),
                   generate_random_keyword(self.alphabet, length)]
        return key

    def encrypt(self, plaintext, key):
        if self.keyword_type in (1, 3, 4):
            pt_alphabet = key[1]
        else:
            pt_alphabet = np.array(list(range(len(self.alphabet))))
        ct_alphabet = self.generate_ct_alphabet(key)
        ciphertext = []
        for i, p in enumerate(plaintext):
            ciphertext.append(ct_alphabet[i % len(key[0])][np.where(pt_alphabet == p)[0][0]])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        if self.keyword_type in (1, 3, 4):
            pt_alphabet = key[1]
        else:
            pt_alphabet = np.array(list(range(len(self.alphabet))))
        ct_alphabet = self.generate_ct_alphabet(key)
        plaintext = []
        for i, c in enumerate(ciphertext):
            plaintext.append(pt_alphabet[np.where(ct_alphabet[i % len(key[0])] == c)[0][0]])
        return np.array(plaintext)

    def generate_ct_alphabet(self, key):
        ct_alphabet = []
        if self.keyword_type == 1:
            for c in key[0]:
                lst = []
                distance = np.where(key[1] == 0)[0][0] - c
                for i in range(len(self.alphabet)):
                    lst.append((i - distance) % len(self.alphabet))
                ct_alphabet.append(lst)
        elif self.keyword_type == 2:
            alphabet = map_text_into_numberspace(
                generate_keyword_alphabet(self.alphabet, map_numbers_into_textspace(key[1], self.alphabet, self.unknown_symbol)),
                self.alphabet, self.unknown_symbol_number)
            for c in key[0]:
                lst = []
                distance = np.where(alphabet == c)[0][0]
                for i in range(len(self.alphabet)):
                    lst.append(alphabet[(i + distance) % len(self.alphabet)])
                ct_alphabet.append(lst)
        elif self.keyword_type == 3:
            alphabet = key[1]
            for c in key[0]:
                lst = []
                distance = np.where(key[1] == c)[0][0]
                for i in range(len(self.alphabet)):
                    lst.append(alphabet[(i + distance) % len(self.alphabet)])
                ct_alphabet.append(lst)
        else:
            alphabet = map_text_into_numberspace(
                generate_keyword_alphabet(self.alphabet, map_numbers_into_textspace(key[2], self.alphabet, self.unknown_symbol)),
                self.alphabet, self.unknown_symbol_number)
            for c in key[0]:
                lst = []
                distance = np.where(alphabet == c)[0][0]
                for i in range(len(self.alphabet)):
                    lst.append(alphabet[(i + distance) % len(self.alphabet)])
                ct_alphabet.append(lst)
        return np.array(ct_alphabet)
