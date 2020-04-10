import random
import sys

sys.path.append("../../../")
from util import text_utils


class Cipher:
    ''' This is the interface of the cipher implementations.'''
    def generate_random_key(self, length):
        if length is None or length <= 0:
            raise ValueError('The length of a key must be greater than 0 and must not be None.')
        if not isinstance(length, int):
            raise ValueError('Length must be of type integer.')
        key = b''
        for i in range(length):
            char = bytes([self.alphabet[int(random.randrange(0, len(self.alphabet) - 1))]])
            key = key + char
        return key

    def encrypt(self, plaintext, key):
        raise Exception('Interface method called')

    def decrypt(self, ciphertext, key):
        raise Exception('Interface method called')

    def filter(self, plaintext, keep_unknown_symbols=False):
        plaintext = plaintext.lower()
        if not keep_unknown_symbols:
            return text_utils.remove_unknown_symbols(plaintext, self.alphabet)
        return plaintext
