from cipherImplementations.progressiveKey import ProgressiveKey
from util.utils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class ProgressiveKeyTest(CipherTestBase):
    cipher = ProgressiveKey(
        CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER, progression_index=1)
    plaintext = b'This cipher can be used with any of the periodics.'
    ciphertext = b'zyihgngbmkjsorjakzmqqmjrtfhbdcnjhjpwxfno'
    decrypted_plaintext = b'thisciphercanbeusedwithanyoftheperiodics'
    key = map_text_into_numberspace(b'grapefruit', cipher.alphabet, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)

    def test1generate_random_key(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for c in key:
            self.assertIn(c, self.cipher.alphabet)

        length = 19
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for c in key:
            self.assertIn(c, self.cipher.alphabet)

        length = 25
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for c in key:
            self.assertIn(c, self.cipher.alphabet)

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
