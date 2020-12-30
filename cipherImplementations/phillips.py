import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword, generate_keyword_alphabet


class Phillips(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        return generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length))

    def encrypt(self, plaintext, key):
        new_key = list(key)
        ciphertext = []
        key_shift_cntr = 0
        for i, p in enumerate(plaintext):
            if i % 5 == 0 and i != 0:
                new_key, key_shift_cntr = self.shift_key(i, key, key_shift_cntr, new_key)
            pos = new_key.index(p)
            pos += 1
            if pos % 5 != 0:
                pos += 5
            pos = pos % len(self.alphabet)
            ciphertext.append(new_key[pos])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        new_key = list(key)
        plaintext = []
        key_shift_cntr = 0
        for i, p in enumerate(ciphertext):
            if i % 5 == 0 and i != 0:
                new_key, key_shift_cntr = self.shift_key(i, key, key_shift_cntr, new_key)
            pos = new_key.index(p)
            if pos % 5 != 0:
                pos -= 5
            pos -= 1
            pos = pos % len(self.alphabet)
            plaintext.append(new_key[pos])
        return np.array(plaintext)

    def shift_key(self, i, key, key_shift_cntr, new_key):
        row_shift = 5 * key_shift_cntr
        tmp = new_key[row_shift:row_shift + 5]
        new_key[row_shift:row_shift + 5] = new_key[row_shift + 5:row_shift + 10]
        new_key[row_shift + 5:row_shift + 10] = tmp
        key_shift_cntr += 1
        if key_shift_cntr == 4:
            key_shift_cntr = 0
        if i % 40 == 0:
            new_key = list(key)
        return new_key, key_shift_cntr

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower().replace(b'j', b'i')
        plaintext = super().filter(bytes(plaintext), keep_unknown_symbols)
        return plaintext
