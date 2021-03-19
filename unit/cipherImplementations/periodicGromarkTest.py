from cipherImplementations.periodicGromark import PeriodicGromark
from util.utils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase
import numpy as np


class PeriodicGromarkTest(CipherTestBase):
    cipher = PeriodicGromark(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Wintry showers will continue for the next few days according to the forecast'
    ciphertext = b'264351rhnaaxnruzbniuarxcrtpatbrligdsvcircvoypvraazzmusreqyevmmurgwtlud4'
    decrypted_plaintext = b'wintryshowerswillcontinueforthenextfewdaysaccordingtotheforecast'
    key = [[2,6,4,3,5,1], np.array([4,21,13,9,17,0]), map_text_into_numberspace(b'ajrxebksygfpvidoumhqwncltz', CipherTestBase.ALPHABET,
           CipherTestBase.UNKNOWN_SYMBOL_NUMBER)]

    def test1generate_random_key(self):
        old_key = self.cipher.alphabet
        old_primer = [2,6,4,3,5,1]
        for i in range(3, 25):
            primer, periodic_key, key = self.cipher.generate_random_key(i)
            self.assertEqual(26, len(key))
            self.assertNotEqual(key, old_key)
            self.assertEqual(i, len(primer))
            self.assertNotEqual(primer, old_primer)
            self.assertEqual(i, len(periodic_key))
            for p in periodic_key:
                self.assertTrue(p % 1 == 0)
            old_primer = primer
            old_key = key

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
