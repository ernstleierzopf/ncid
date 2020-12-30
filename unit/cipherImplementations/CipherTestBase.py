import unittest
from util.textUtils import map_text_into_numberspace, map_numbers_into_textspace
import random
import numpy as np
from cipherImplementations.cipher import OUTPUT_ALPHABET


class CipherTestBase(unittest.TestCase):
    ALPHABET = b'abcdefghijklmnopqrstuvwxyz'
    UNKNOWN_SYMBOL = b'?'
    UNKNOWN_SYMBOL_NUMBER = 90

    def run_test1generate_random_key_allowed_length(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for c in key:
            self.assertIn(c, self.ALPHABET)

        length = 19
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for c in key:
            self.assertIn(c, self.ALPHABET)

        length = 150
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for c in key:
            self.assertIn(c, self.ALPHABET)

    def run_test1generate_random_list_of_unique_digits(self):
        for i in range(1, 100):
            length = random.randint(0, i)
            key = self.cipher.generate_random_key(length)
            self.assertEqual(length, len(np.unique(key)))

    def run_test1generate_random_alphabet(self):
        old_key = self.cipher.alphabet
        for _ in range(0, 100):
            key = self.cipher.generate_random_key()
            self.assertEqual(26, len(key))
            self.assertNotEqual(key, old_key)
            old_key = key

    def run_test2generate_random_key_wrong_length_parameter(self):
        self.assertRaises(ValueError, self.cipher.generate_random_key, 0)
        self.assertRaises(ValueError, self.cipher.generate_random_key, -1)
        self.assertRaises(ValueError, self.cipher.generate_random_key, 1.55)
        self.assertRaises(ValueError, self.cipher.generate_random_key, None)

    def run_test3filter_keep_unknown_symbols(self):
        self.assertEqual(self.cipher.filter(self.plaintext, keep_unknown_symbols=True), self.plaintext.lower())

    def run_test4filter_delete_unknown_symbols(self):
        self.assertEqual(self.cipher.filter(self.plaintext, keep_unknown_symbols=False), self.decrypted_plaintext)

    def run_test5encrypt(self):
        alph = b'' + OUTPUT_ALPHABET
        if b'j' not in self.cipher.alphabet:
            alph = alph.replace(b'j', b'')
        if b'x' not in self.cipher.alphabet:
            alph = alph.replace(b'x', b'')
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=False)
        plaintext_numbers = map_text_into_numberspace(plaintext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
        ciphertext = map_numbers_into_textspace(ciphertext_numbers, alph, self.UNKNOWN_SYMBOL)
        self.assertEqual(self.ciphertext, ciphertext)

    def run_test6decrypt(self):
        alph = b'' + OUTPUT_ALPHABET
        if b'j' not in self.cipher.alphabet:
            alph = alph.replace(b'j', b'')
        if b'x' not in self.cipher.alphabet:
            alph = alph.replace(b'x', b'')
        ciphertext_numbers = map_text_into_numberspace(self.ciphertext, alph, self.UNKNOWN_SYMBOL_NUMBER)
        plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key)
        plaintext = map_numbers_into_textspace(plaintext_numbers, self.cipher.alphabet, self.UNKNOWN_SYMBOL)
        self.assertEqual(self.decrypted_plaintext, plaintext)


if __name__ == '__main__':
    unittest.main()
