from cipherImplementations.cipher import Cipher
from cipherImplementations.vigenere import Vigenere


class RunningKey(Cipher):
    """This implementation takes the ciphertext off in rows."""

    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number
        self.vigenere = Vigenere(alphabet, unknown_symbol, unknown_symbol_number)

    def generate_random_key(self, length=None):
        return None

    def encrypt(self, plaintext, key):
        split = int(len(plaintext) / 2) + (len(plaintext) % 2 > 0)
        key = plaintext[:split]
        plaintext = plaintext[split:]
        return self.vigenere.encrypt(plaintext, key)

    def decrypt(self, ciphertext, key):
        raise Exception('This cipher can not be decrypted!')
