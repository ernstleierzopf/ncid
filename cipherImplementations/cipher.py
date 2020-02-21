from cipherImplementations import monoalphabetic_substitution
''' This is the interface of the cipher implementations.'''

CIPHER_TYPES = ['monoalphabetic_substitution', 'vigenere', 'columnar_transposition', 'playfair', 'hill']
CIPHER_IMPLEMENTATIONS = [monoalphabetic_substitution]
KEY_LENGTHS = [None]
MTC3 = 'mtc3'

class Cipher:
    def generate_random_key(self, alphabet, length):
        raise Exception('Interface method called')

    def encrypt (self, plaintext, key):
        raise Exception('Interface method called')