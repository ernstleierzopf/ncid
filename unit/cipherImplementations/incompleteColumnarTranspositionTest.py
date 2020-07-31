from cipherImplementations.columnarTransposition import ColumnarTransposition
from unit.cipherImplementations.CipherTestBase import CipherTestBase
import numpy as np


class IncompleteColumnarTranspositionTest(CipherTestBase):
    cipher = ColumnarTransposition(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER,
                                   fill_blocks=False)
    plaintext = b'Unfilled block'
    key = np.array([2,0,1])
    ciphertext = b'nldoflbcuielk'
    decrypted_plaintext = b'unfilledblock'

    def test1generate_random_key_allowed_length(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        alph = list(range(length))
        for c in key:
            self.assertIn(c, alph)
            alph.remove(c)
        self.assertEqual(alph, [])

        length = 19
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        alph = list(range(length))
        for c in key:
            self.assertIn(c, alph)
            alph.remove(c)
        self.assertEqual(alph, [])

        length = 150
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        alph = list(range(length))
        for c in key:
            self.assertIn(c, alph)
            alph.remove(c)
        self.assertEqual(alph, [])

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.assertEqual(self.cipher.filter(self.plaintext, keep_unknown_symbols=False), self.decrypted_plaintext)

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()