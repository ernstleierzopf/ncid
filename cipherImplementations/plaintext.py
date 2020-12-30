from cipherImplementations.cipher import Cipher


class Plaintext(Cipher):
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length=None):
        return None

    def encrypt(self, plaintext, key):
        return plaintext

    def decrypt(self, ciphertext, key):
        return ciphertext
