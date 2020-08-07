import numpy as np
from cipherImplementations.redefence import Redefence
import random
from collections import deque


class Railfence(Redefence):
    """This implementation takes the ciphertext off in rows."""
    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number):
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        if length is None or length <= 0:
            raise ValueError('The length of a key must be greater than 0 and must not be None.')
        if not isinstance(length, int):
            raise ValueError('Length must be of type integer.')
        return [np.array([i for i in range(length)]), random.randint(0, 15)]