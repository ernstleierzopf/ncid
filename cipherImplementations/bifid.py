from cipherImplementations.cipher import Cipher, generate_random_keyword, generate_keyword_alphabet
from cipherImplementations.polybius import Polybius
import numpy as np


class Bifid(Cipher):
    """Adapted implementation from https://github.com/tigertv/secretpy"""

    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        return [generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length)), length]

    def encrypt(self, plaintext, key):
        pt = []
        for p in plaintext:
            pt.append(np.where(key[0] == p)[0][0])
        plaintext = pt
        __polybius = Polybius(key[0], self.unknown_symbol, self.unknown_symbol_number)
        if not key[1] > 0:
            key[1] = len(plaintext)
        code = __polybius.encrypt(plaintext, list(range(len(self.alphabet))))
        even = code[::2]
        odd = code[1::2]
        ret = []
        for i in range(0, len(even), key[1]):
            ret += even[i:i + key[1]] + odd[i:i + key[1]]
        ct = __polybius.decrypt(ret, list(range(len(self.alphabet))))
        ciphertext = []
        for c in ct:
            ciphertext.append(key[0][c])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        ct = []
        for c in ciphertext:
            ct.append(np.where(key[0] == c)[0][0])
        ciphertext = ct
        __polybius = Polybius(key[0], self.unknown_symbol, self.unknown_symbol_number)
        if not key[1] > 0:
            key[1] = len(ciphertext)
        code = __polybius.encrypt(ciphertext, list(range(len(self.alphabet))))
        even = ''
        odd = ''
        rem = len(code) % (key[1] << 1)
        for i in range(0, len(code) - rem, key[1] << 1):
            ikey = i + key[1]
            even += code[i:ikey]
            odd += code[ikey:ikey + key[1]]

        even += code[-rem:-(rem >> 1)]
        odd += code[-(rem >> 1):]

        code = []
        for i in range(len(even)):
            code += even[i] + odd[i]
        pt = __polybius.decrypt(code, list(range(len(self.alphabet))))
        plaintext = []
        for p in pt:
            plaintext.append(key[0][p])
        return np.array(plaintext)

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower().replace(b'j', b'i')
        plaintext = super().filter(bytes(plaintext), keep_unknown_symbols)
        return plaintext
