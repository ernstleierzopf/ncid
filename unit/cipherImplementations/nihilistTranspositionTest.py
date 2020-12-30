from cipherImplementations.nihilistTransposition import NihilistTransposition
from unit.cipherImplementations.CipherTestBase import CipherTestBase
import numpy as np


class NihilistTranspositionTest(CipherTestBase):
    cipher = NihilistTransposition(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'square needed here'
    ciphertext = b'eqdersehnuereade'
    decrypted_plaintext = b'squareneededhere'
    key = np.array([1,0,2,3])

    def test1generate_random_key_allowed_length(self):
        for i in range(1, 100):
            key = self.cipher.generate_random_key(i)
            self.assertEqual(i, len(np.unique(key)))

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()
