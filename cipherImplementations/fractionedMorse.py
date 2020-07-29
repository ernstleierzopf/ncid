from cipherImplementations.cipher import Cipher
from util.textUtils import remove_unknown_symbols
import random
import numpy as np


class FractionedMorse(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number
        # morse code in alphabetical order
        self.morse_codes = ['.-', '-...', '-.-.', '-..', '.', '..-.', '--.', '....', '..', '.---', '-.-', '.-..', '--', '-.', '---', '.--.',
                            '--.-', '.-.', '...', '-', '..-', '...-', '.--', '-..-', '-.--', '--..']
        # key letter substitution
        self.key_morse = ['...', '..-', '..x', '.-.', '.--', '.-x', '.x.', '.x-', '.xx', '-..', '-.-', '-.x', '--.', '---', '--x', '-x.',
                          '-x-', '-xx', 'x..', 'x.-', 'x.x', 'x-.', 'x--', 'x-x', 'xx.', 'xx-']

    def generate_random_key(self, length=None):
        alphabet2 = b'' + self.alphabet.replace(b' ', b'')
        key = b''
        for _ in range(len(self.alphabet.replace(b' ', b''))):
            position = int(random.randrange(0, len(alphabet2)))
            char = bytes([alphabet2[position]])
            key = key + char
            alphabet2 = alphabet2.replace(char, b'')
        return key

    def encrypt(self, plaintext, key):
        morse_code = ''
        for c in plaintext:
            if c == 26:
                morse_code += 'x'
                continue
            morse_code += self.morse_codes[c] + 'x'
        morse_code += 'x'

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
                plaintext.append(self.morse_codes.index(tmp))
                tmp = 'x'
        return np.array(plaintext[:-1])

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower()
        if not keep_unknown_symbols:
            return remove_unknown_symbols(plaintext, self.alphabet + b' ')
        return plaintext