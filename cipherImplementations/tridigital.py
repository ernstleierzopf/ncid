import numpy as np
from cipherImplementations.cipher import Cipher, generate_random_keyword, generate_random_list_of_unique_digits, generate_keyword_alphabet,\
    OUTPUT_ALPHABET


class Tridigital(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        return [generate_random_list_of_unique_digits(10), generate_keyword_alphabet(
            self.alphabet.replace(b' ', b''), generate_random_keyword(self.alphabet.replace(b' ', b''), length))]

    def encrypt(self, plaintext, key):
        ciphertext = []
        for p in plaintext:
            if p == self.alphabet.index(b' '):
                ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(key[0][9]), encoding='utf-8')))
                continue
            ciphertext.append(OUTPUT_ALPHABET.index(bytes(str(key[0][np.where(key[1] == p)[0][0] % 9]), encoding='utf-8')))
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        raise Exception("Decryption of the Tridigital cipher is not possible.")
