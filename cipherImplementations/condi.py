from cipherImplementations.cipher import Cipher
import random


class Condi(Cipher):
    """Adapted implementation from https://github.com/tigertv/secretpy"""
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        alphabet2 = b'' + self.alphabet
        key = b''
        for _ in range(len(self.alphabet)):
            position = int(random.randrange(0, len(alphabet2)))
            char = bytes([alphabet2[position]])
            key = key + char
            alphabet2 = alphabet2.replace(char, b'')
        return [key, random.randint(0, len(self.alphabet)-1)]

    def encrypt(self, plaintext, key):
        ciphertext = []
        alphabet = list(key[0])
        offset = key[1]
        for c in plaintext:
            #print(offset)
            offset = (alphabet.index(c) + offset) % len(alphabet)
            ciphertext.append(alphabet[offset])
            offset = alphabet.index(c) + 1
        return ciphertext

    def decrypt(self, ciphertext, key):
        plaintext = []
        ciphertext = list(ciphertext)
        alphabet = list(key[0])
        offset = key[1]
        for i in range(0, len(ciphertext), 1):
            plaintext.append(alphabet[(alphabet.index(ciphertext[i]) - offset) % len(alphabet)])
            offset = (alphabet.index(ciphertext[i]) - offset + 1) % len(alphabet)
        return plaintext