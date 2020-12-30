from cipherImplementations.cipher import Cipher, generate_keyword_alphabet, generate_random_keyword, OUTPUT_ALPHABET
from cipherImplementations.polybius import Polybius
import numpy as np


class NihilistSubstitution(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        if length is None or length >= len(self.alphabet):
            raise ValueError('The length must not be greater than the length of the alphabet.')
        keyword = generate_random_keyword(self.alphabet, length)
        key = generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length))
        return [keyword, key]

    def encrypt(self, plaintext, key):
        pt = []
        for p in plaintext:
            pt.append(np.where(key[1] == p)[0][0])
        plaintext = pt
        __polybius = Polybius(key[1], self.unknown_symbol, self.unknown_symbol_number)
        code = __polybius.encrypt(plaintext, list(range(len(self.alphabet))))
        kw = []
        for k in key[0]:
            kw.append(np.where(key[1] == k)[0][0])
        keyword_code = __polybius.encrypt(kw, list(range(len(self.alphabet))))
        kw = []
        for i in range(0, len(keyword_code), 2):
            kw.append(int(keyword_code[i:i+2]))
        ct = []
        for i in range(0, len(code), 2):
            ct.append(int(code[i:i+2]))
        for i in range(len(ct)):
            ct[i] = ct[i] + kw[i % len(kw)]
            if ct[i] >= 100:
                ct[i] = ct[i] % 100

        ciphertext = []
        for c in ct:
            ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(int(c / 10)), 'utf-8')))
            ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(int(c % 10)), 'utf-8')))
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        ciphertext = np.array([int(bytes([OUTPUT_ALPHABET[c]])) for c in ciphertext])
        __polybius = Polybius(key[1], self.unknown_symbol, self.unknown_symbol_number)
        kw = []
        for k in key[0]:
            kw.append(np.where(key[1] == k)[0][0])
        keyword_code = __polybius.encrypt(kw, list(range(len(self.alphabet))))
        kw = []
        for i in range(0, len(keyword_code), 2):
            kw.append(int(keyword_code[i:i + 2]))
        ct = []
        for i in range(0, len(ciphertext), 2):
            c = ciphertext[i] * 10 + ciphertext[i + 1]
            if c <= 10:
                c += 100
            ct.append(c)

        for i in range(len(ct)):
            ct[i] = ct[i] - kw[i % len(kw)]

        ciphertext = ''
        for c in ct:
            ciphertext += str(c)
        plaintext = __polybius.decrypt(ciphertext, key[1])
        return np.array(plaintext)
