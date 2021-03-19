from cipherImplementations.monomeDinome import MonomeDinome
from util.utils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase
import numpy as np


class MonomeDinomeTest(CipherTestBase):
    cipher = MonomeDinome(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'highfrequencykeysshortenciphertext'
    ciphertext = b'6006760627539325168346553444608795168038605795359'
    decrypted_plaintext = b'highfrequencykeysshortenciphertext'
    key = [np.array([6,3,1,8,9,2,7,0,5,4]), map_text_into_numberspace(
        b'notariesbcdfghklmpquvwxy', cipher.alphabet, cipher.unknown_symbol_number)]

    def test1generate_random_key(self):
        old_key = self.cipher.alphabet
        old_numbers = self.key[0]
        for _ in range(0, 100):
            numbers, key = self.cipher.generate_random_key()
            self.assertEqual(26, len(key))
            self.assertNotEqual(key, old_key)
            self.assertEqual(10, len(numbers))
            self.assertFalse(np.array_equal(numbers, old_numbers))
            old_key = key
            old_numbers = numbers

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()
