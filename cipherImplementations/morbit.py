from cipherImplementations.cipher import Cipher, generate_random_list_of_unique_digits, OUTPUT_ALPHABET
from util.utils import remove_unknown_symbols, encrypt_morse, decrypt_morse, morse_codes
import numpy as np


class Morbit(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number
        # key letter substitution
        self.key_morse = ['..', '.-', '.x', '-.', '--', '-x', 'x.', 'x-', 'xx']

    def generate_random_key(self, length=None):
        return generate_random_list_of_unique_digits(9)

    def encrypt(self, plaintext, key):
        morse_code = encrypt_morse(plaintext)

        ciphertext = []
        for i in range(0, len(morse_code) - 1, 2):
            value = key[self.key_morse.index(morse_code[i:i+2])]
            ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(value), 'utf-8')))
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        ciphertext = np.array([int(bytes([OUTPUT_ALPHABET[c]])) for c in ciphertext])
        morse_code = decrypt_morse(ciphertext, self.key_morse, key)

        plaintext = []
        tmp = ''
        for c in morse_code:
            if c == 'x' and tmp == 'x':
                tmp = ''
                plaintext.append(OUTPUT_ALPHABET.index(b' '))
            elif c != 'x':
                if tmp == 'x':
                    tmp = ''
                tmp += c
            else:
                plaintext.append(morse_codes.index(tmp))
                tmp = 'x'
        return np.array(plaintext[:-1])

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower()
        if not keep_unknown_symbols:
            return remove_unknown_symbols(plaintext, self.alphabet + b' ')
        return plaintext
