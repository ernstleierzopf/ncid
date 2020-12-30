import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword, generate_keyword_alphabet


class Trifid(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number
        self.key_dict = {
            0: [1,1,1], 1: [1,1,2], 2: [1,1,3], 3: [1,2,1], 4: [1,2,2], 5: [1,2,3], 6: [1,3,1], 7: [1,3,2], 8: [1,3,3], 9: [2,1,1],
            10: [2,1,2], 11: [2,1,3], 12: [2,2,1], 13: [2,2,2], 14: [2,2,3], 15: [2,3,1], 16: [2,3,2], 17: [2,3,3], 18: [3,1,1],
            19: [3,1,2], 20: [3,1,3], 21: [3,2,1], 22: [3,2,2], 23: [3,2,3], 24: [3,3,1], 25: [3,3,2], 26: [3,3,3]
        }

    def generate_random_key(self, length):
        return [generate_keyword_alphabet(
            self.alphabet.replace(b'#', b''), generate_random_keyword(self.alphabet.replace(b'#', b''), length)) + b'#', length]

    def encrypt(self, plaintext, key):
        ciphertext = []
        mapping = [[] for _ in range(3)]
        for p in plaintext:
            val = self.key_dict[np.where(key[0] == p)[0][0]]
            mapping[0].append(val[0])
            mapping[1].append(val[1])
            mapping[2].append(val[2])
        tmp = []
        end_shift = 0
        for i in range(len(plaintext) * 3):
            shift = int(i / (key[1] * 3)) * key[1]
            row = int(i / key[1]) % 3
            column = i % key[1]
            if column + shift + end_shift >= len(plaintext):
                if len(plaintext) - shift - column + 1 > end_shift:
                    end_shift = len(plaintext) - shift - column + 1
                row += 1
                column = (column + shift + end_shift) % len(plaintext) - end_shift - 1
            if end_shift > 0:
                column += end_shift
            if shift + column >= len(plaintext):
                if (shift + column + end_shift) % len(plaintext) <= end_shift:
                    end_shift += 1
                else:
                    column -= 1
                row += 1
                column = (column + shift) % len(plaintext)
            tmp.append(mapping[row][shift + column])
            if len(tmp) == 3:
                ct = key[0][list(self.key_dict.values()).index(tmp)]
                if ct == 26:
                    ct += 1
                ciphertext.append(ct)
                tmp = []
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        plaintext = []
        mapping = []
        for c in ciphertext:
            val = self.key_dict[np.where(key[0] == c)[0][0]]
            mapping.append(val[0])
            mapping.append(val[1])
            mapping.append(val[2])
        for i in range(len(ciphertext) - len(ciphertext) % key[1]):
            shift = int(i / (key[1])) * key[1] * 3
            j = i % key[1]
            tmp = [mapping[j+shift], mapping[j+shift+key[1]], mapping[j+shift+2*key[1]]]
            plaintext.append(key[0][list(self.key_dict.values()).index(tmp)])

        end_shift = len(ciphertext) % key[1]
        for i in range(len(ciphertext) - len(ciphertext) % key[1], len(ciphertext), 1):
            shift = int(i / key[1]) * key[1] * 3
            j = i % key[1]
            tmp = [mapping[j+shift], mapping[j+shift+end_shift], mapping[j+shift+2*end_shift]]
            plaintext.append(key[0][list(self.key_dict.values()).index(tmp)])
        return np.array(plaintext)
