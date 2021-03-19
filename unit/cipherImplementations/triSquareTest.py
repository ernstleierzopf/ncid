from cipherImplementations.triSquare import TriSquare
from util.utils import map_text_into_numberspace, map_numbers_into_textspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase
import numpy as np


class TriSquareTest(CipherTestBase):
    cipher = TriSquare(CipherTestBase.ALPHABET.replace(b'j', b''), CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'three keysquares used'
    ciphertext = b'rhlqxrlxoevzbatxserxddiuaaabfz'
    decrypted_plaintext = b'threekeysquaresusedx'
    key = [map_text_into_numberspace(b'nsfmuoagpwvbhqxeciryldktz', cipher.alphabet, CipherTestBase.UNKNOWN_SYMBOL_NUMBER),
           map_text_into_numberspace(b'readingbcfhklmopqstuvwxyz', cipher.alphabet, CipherTestBase.UNKNOWN_SYMBOL_NUMBER),
           map_text_into_numberspace(b'pastinoqrmlyzuekxwvbhgfdc', cipher.alphabet, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)]

    def test1generate_random_key(self):
        length = 5
        key1, key2, key3 = self.cipher.generate_random_key(length)
        self.assertEqual(len(key1), len(self.cipher.alphabet))
        self.assertEqual(len(key2), len(self.cipher.alphabet))
        self.assertEqual(len(key3), len(self.cipher.alphabet))
        alphabet2 = b'' + self.cipher.alphabet
        for c in key1:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')
        alphabet2 = b'' + self.cipher.alphabet
        for c in key2:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')
        alphabet2 = b'' + self.cipher.alphabet
        for c in key3:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')

        length = 19
        key1, key2, key3 = self.cipher.generate_random_key(length)
        self.assertEqual(len(key1), len(self.cipher.alphabet))
        self.assertEqual(len(key2), len(self.cipher.alphabet))
        self.assertEqual(len(key3), len(self.cipher.alphabet))
        alphabet2 = b'' + self.cipher.alphabet
        for c in key1:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')
        alphabet2 = b'' + self.cipher.alphabet
        for c in key2:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')
        alphabet2 = b'' + self.cipher.alphabet
        for c in key3:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')

        length = 25
        key1, key2, key3 = self.cipher.generate_random_key(length)
        self.assertEqual(len(key1), len(self.cipher.alphabet))
        self.assertEqual(len(key2), len(self.cipher.alphabet))
        self.assertEqual(len(key3), len(self.cipher.alphabet))
        alphabet2 = b'' + self.cipher.alphabet
        for c in key1:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')
        alphabet2 = b'' + self.cipher.alphabet
        for c in key2:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')
        alphabet2 = b'' + self.cipher.alphabet
        for c in key3:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.assertEqual(self.cipher.filter(self.plaintext, keep_unknown_symbols=False), self.decrypted_plaintext.replace(b'x', b''))

    def test5encrypt(self):
        alph = b'' + self.cipher.alphabet
        if b'j' not in self.cipher.alphabet:
            alph = alph.replace(b'j', b'')
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=False)
        plaintext_numbers = map_text_into_numberspace(plaintext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
        ciphertext = map_numbers_into_textspace(ciphertext_numbers, alph, self.UNKNOWN_SYMBOL)
        for i in range(0, len(self.ciphertext), 3):
            self.assertEqual(np.where(self.key[0] == alph.index(self.ciphertext[i]))[0][0] % 5,
                             np.where(self.key[0] == alph.index(ciphertext[i]))[0][0] % 5)
            self.assertEqual(int(np.where(self.key[2] == alph.index(self.ciphertext[i+1]))[0][0] / 5),
                             int(np.where(self.key[2] == alph.index(ciphertext[i+1]))[0][0] / 5))
            self.assertEqual(np.where(self.key[2] == alph.index(self.ciphertext[i+1]))[0][0] % 5,
                             np.where(self.key[2] == alph.index(ciphertext[i+1]))[0][0] % 5)
            self.assertEqual(int(np.where(self.key[1] == alph.index(self.ciphertext[i+2]))[0][0] / 5),
                             int(np.where(self.key[1] == alph.index(ciphertext[i+2]))[0][0] / 5))

    def test6decrypt(self):
        self.run_test6decrypt()
