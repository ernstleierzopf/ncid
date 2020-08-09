from cipherImplementations.tridigital import Tridigital
from util.textUtils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase
import numpy as np


class TridigitalTest(CipherTestBase):
    cipher = Tridigital(CipherTestBase.ALPHABET + b' ', CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'the ides of march'
    ciphertext = b'03095607958910773'
    decrypted_plaintext = b'the ides of march'
    key = [np.array([6,7,0,3,5,2,8,1,4,9]),
           map_text_into_numberspace(b'dragonflybcehijkmpqstuvwxz', cipher.alphabet, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)]

    def test1generate_random_key(self):
        length = 5
        numbers, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(CipherTestBase.ALPHABET))
        alphabet2 = b'' + CipherTestBase.ALPHABET
        for c in key:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')
        self.assertEqual(len(numbers), 10)
        self.assertEqual(len(set(numbers)), 10)
        for i in numbers:
            self.assertIsInstance(i, np.int_)

        length = 19
        numbers, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(CipherTestBase.ALPHABET))
        alphabet2 = b'' + CipherTestBase.ALPHABET
        for c in key:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')
        self.assertEqual(len(numbers), 10)
        self.assertEqual(len(set(numbers)), 10)
        for i in numbers:
            self.assertIsInstance(i, np.int_)

        length = 25
        numbers, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(CipherTestBase.ALPHABET))
        alphabet2 = b'' + CipherTestBase.ALPHABET
        for c in key:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')
        self.assertEqual(len(numbers), 10)
        self.assertEqual(len(set(numbers)), 10)
        for i in numbers:
            self.assertIsInstance(i, np.int_)

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    # def test6decrypt(self):
    #     self.run_test6decrypt()