import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword, generate_keyword_alphabet


class Ragbaby(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        return generate_keyword_alphabet(self.alphabet, generate_random_keyword(self.alphabet, length))

    def encrypt(self, plaintext, key):
        ciphertext = []
        words = extract_word_list(plaintext)
        cntr = 0
        for word in words:
            for i in range(len(word)):
                ciphertext.append(key[(np.where(key == word[i])[0][0] + 1 + cntr + i) % len(key)])
            ciphertext.append(24)
            cntr += 1
        return np.array(ciphertext[:-1])

    def decrypt(self, ciphertext, key):
        plaintext = []
        words = extract_word_list(ciphertext)
        cntr = 0
        for word in words:
            for i in range(len(word)):
                plaintext.append(key[(np.where(key == word[i])[0][0] - 1 - cntr - i) % len(key)])
            plaintext.append(24)
            cntr += 1
        return np.array(plaintext[:-1])

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower().replace(b'j', b'i')
        plaintext = plaintext.lower().replace(b'x', b'w')
        plaintext = super().filter(bytes(plaintext), keep_unknown_symbols)
        return plaintext


def extract_word_list(text):
    word_indices = np.where(text == 24)[0]
    words = []
    old_indice = 0
    for i in word_indices:
        words.append(text[old_indice:i])
        old_indice = i + 1
    words.append(text[old_indice:len(text)])
    return words
