from cipherImplementations.cipher import Cipher, generate_random_keyword
import numpy as np


class Autokey(Cipher):
    """Adapted implementation from https://github.com/nishimehta/AutokeyCryptanalysis"""

    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        return generate_random_keyword(self.alphabet, length)

    def encrypt(self, plaintext, key):
        return self.__enc_dec(plaintext, key, 'encrypt')

    def decrypt(self, ciphertext, key):
        return self.__enc_dec(ciphertext, key, 'decrypt')

    def __enc_dec(self, message, key, mode):
        cipher = []
        k_index = 0
        # here the key has to be converted back to a list as it must be extended.
        key = list(key)
        for i in message:
            text = i
            if mode == 'encrypt':
                text += key[k_index]
                key.append(i)  # add current char to keystream

            elif mode == 'decrypt':
                text -= key[k_index]
                key.append(text)  # add current char to keystream
            text %= len(self.alphabet)
            k_index += 1
            cipher.append(text)
        return np.array(cipher)
