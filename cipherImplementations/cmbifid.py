from cipherImplementations.bifid import Bifid
from cipherImplementations.cipher import generate_random_keyword, generate_keyword_alphabet
from cipherImplementations.polybius import Polybius
import numpy as np


class CMBifid(Bifid):
    """Adapted implementation from https://github.com/tigertv/secretpy"""

    def generate_random_key(self, length):
        key1 = generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length))
        key2 = generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length), vertical=True)
        return [length, key1, key2]

    def encrypt(self, plaintext, key):
        pt = []
        for p in plaintext:
            pt.append(np.where(key[1] == p)[0][0])
        plaintext = pt
        __polybius = Polybius(key[1], self.unknown_symbol, self.unknown_symbol_number)
        if not key[0] > 0:
            key[0] = len(plaintext)
        code = __polybius.encrypt(plaintext, list(range(len(self.alphabet))))
        even = code[::2]
        odd = code[1::2]
        ret = []
        for i in range(0, len(even), key[0]):
            ret += even[i:i + key[0]] + odd[i:i + key[0]]
        ct = __polybius.decrypt(ret, list(range(len(self.alphabet))))
        ciphertext = []
        for c in ct:
            ciphertext.append(key[2][c])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        ct = []
        for c in ciphertext:
            ct.append(np.where(key[2] == c)[0][0])
        ciphertext = ct
        __polybius = Polybius(key[1], self.unknown_symbol, self.unknown_symbol_number)
        if not key[0] > 0:
            key[0] = len(ciphertext)
        code = __polybius.encrypt(ciphertext, list(range(len(self.alphabet))))
        even = ''
        odd = ''
        rem = len(code) % (key[0] << 1)
        for i in range(0, len(code) - rem, key[0] << 1):
            ikey = i + key[0]
            even += code[i:ikey]
            odd += code[ikey:ikey + key[0]]

        even += code[-rem:-(rem >> 1)]
        odd += code[-(rem >> 1):]

        code = []
        for i in range(len(even)):
            code += even[i] + odd[i]
        pt = __polybius.decrypt(code, list(range(len(self.alphabet))))
        plaintext = []
        for p in pt:
            plaintext.append(key[1][p])
        return np.array(plaintext)
