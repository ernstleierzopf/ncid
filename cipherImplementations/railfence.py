import numpy as np
from cipherImplementations.redefence import Redefence
import random


class Railfence(Redefence):
    """This implementation takes the ciphertext off in rows."""

    def generate_random_key(self, length):
        if length is None or length <= 0:
            raise ValueError('The length of a key must be greater than 0 and must not be None.')
        if not isinstance(length, int):
            raise ValueError('Length must be of type integer.')
        return [np.array(list(range(length))), random.randint(0, 15)]
