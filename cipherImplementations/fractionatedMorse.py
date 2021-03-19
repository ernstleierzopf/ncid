from cipherImplementations.cipher import Cipher, generate_keyword_alphabet, generate_random_keyword
from util.utils import remove_unknown_symbols, encrypt_morse, morse_codes
import numpy as np


class FractionatedMorse(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number
        # key letter substitution
        self.key_morse = ['...', '..-', '..x', '.-.', '.--', '.-x', '.x.', '.x-', '.xx', '-..', '-.-', '-.x', '--.', '---', '--x', '-x.',
                          '-x-', '-xx', 'x..', 'x.-', 'x.x', 'x-.', 'x--', 'x-x', 'xx.', 'xx-']

    def generate_random_key(self, length):
        alphabet = self.alphabet.replace(b' ', b'')
        return generate_keyword_alphabet(alphabet, generate_random_keyword(alphabet, length))

    def encrypt(self, plaintext, key):
        morse_code = encrypt_morse(plaintext)

        ciphertext = []
        for i in range(0, len(morse_code) - 2, 3):
            ciphertext.append(key[self.key_morse.index(morse_code[i:i+3])])
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        morse_code = ''
        for c in ciphertext:
            morse_code += self.key_morse[np.where(key == c)[0][0]]

        plaintext = []
        tmp = ''
        for c in morse_code:
            if c == 'x' and tmp == 'x':
                tmp = ''
                plaintext.append(26)
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
