from cipherImplementations.morbit import Morbit
from unit.cipherImplementations.CipherTestBase import CipherTestBase
import numpy as np


class MorbitTest(CipherTestBase):
    cipher = Morbit(CipherTestBase.ALPHABET + b' ', CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Once upon a time.'
    ciphertext = b'27435881512827465679378'
    decrypted_plaintext = b'once upon a time'
    key = np.array([9,5,8,4,2,7,1,3,6])

    def test1generate_random_key(self):
        for _ in range(0, 100):
            key = self.cipher.generate_random_key()
            self.assertEqual(9, len(key))

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()