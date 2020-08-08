from cipherImplementations.swagman import Swagman
from unit.cipherImplementations.CipherTestBase import CipherTestBase
import numpy as np


class SwagmanTest(CipherTestBase):
    cipher = Swagman(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Don\'t be afraid to take a big leap if one is indicated. You cannot cross a river or a chasm in two small jumps.'
    ciphertext = b'endscmordaniboisictnastgbltewaoareefsaidvpyrmoeaiafuilrldocotjnraaenouncmitsoaphskati'
    decrypted_plaintext = b'dontbeafraidtotakeabigleapifoneisindicatedyoucannotcrossariverorachasmintwosmalljumps'
    key = np.array([[2,1,0,3,4], [0,4,2,1,3], [1,3,4,2,0], [4,2,3,0,1], [3,0,1,4,2]])

    def test1generate_random_key(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for row in key:
            self.assertEqual(length, len(set(row)))
        for column in range(len(key)):
            column_values = []
            for row in range(len(key)):
                column_values.append(key[row][column])
            self.assertEqual(length, len(set(column_values)))

        length = 7
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for row in key:
            self.assertEqual(length, len(set(row)))
        for column in range(len(key)):
            column_values = []
            for row in range(len(key)):
                column_values.append(key[row][column])
            self.assertEqual(length, len(set(column_values)))

        length = 10
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for row in key:
            self.assertEqual(length, len(set(row)))
        for column in range(len(key)):
            column_values = []
            for row in range(len(key)):
                column_values.append(key[row][column])
            self.assertEqual(length, len(set(column_values)))

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