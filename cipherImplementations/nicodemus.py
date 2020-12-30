import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword
from cipherImplementations.vigenere import Vigenere


class Nicodemus(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number
        self.vigenere = Vigenere(alphabet, unknown_symbol, unknown_symbol_number)

    def generate_random_key(self, length):
        if length is None or length <= 0 or length >= len(self.alphabet):
            raise ValueError('The length of a key must be greater than 0 and smaller than the length of the alphabet and must not be None.')
        if not isinstance(length, int):
            raise ValueError('Length must be of type integer.')
        return generate_random_keyword(self.alphabet, length, unique=True)

    def encrypt(self, plaintext, key):
        transposition_key = np.argsort(key)
        ciphertext = []
        splits = int(len(plaintext) / (len(key) * 5)) + (len(plaintext) % (len(key) * 5) > 0)
        for start in range(len(transposition_key) * splits):
            position = np.where(transposition_key == start % len(key))[0][0] + int(start / len(key)) * 5 * len(key)
            cnt = 0
            tmp = []
            while position < len(plaintext) and cnt < 5:
                p = plaintext[position]
                if p > len(self.alphabet):
                    ciphertext.append(self.unknown_symbol_number)
                    position = position + len(key)
                    continue
                tmp.append(p)
                position = position + len(key)
                cnt += 1
            ciphertext += list(self.vigenere.encrypt(tmp, [key[np.where(transposition_key == start % len(key))[0][0]]]))
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        transposition_key = np.argsort(key)
        plaintext = [b'']*len(ciphertext)
        i = 0
        splits = int(len(plaintext) / (len(key) * 5)) + (len(plaintext) % (len(key) * 5) > 0)
        for start in range(len(transposition_key) * splits):
            position = np.where(transposition_key == start % len(key))[0][0] + int(start / len(key)) * 5 * len(key)
            cnt = 0
            while position < len(plaintext) and cnt < 5:
                c = ciphertext[i]
                i += 1
                if c > len(self.alphabet):
                    plaintext[position] = self.unknown_symbol_number
                    position = position + len(key)
                    continue
                plaintext[position] = self.vigenere.decrypt([c], [key[np.where(transposition_key == start % len(key))[0][0]]])[0]
                position = position + len(key)
                cnt += 1
        return np.array(plaintext)
