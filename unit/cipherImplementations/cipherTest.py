import unittest
from cipherImplementations.cipher import Cipher

class CipherTest(unittest.TestCase):
    ALPHABET = b'abcdefghijklmnopqrstuvwxyz'
    UNKNOWN_SYMBOL = b'?'
    UNKNOWN_SYMBOL_NUMBER = 90
    cipher = Cipher()
    cipher.alphabet = ALPHABET
    plaintext = b'This is a plaintext with special characters!%%xy<'
    filtered_plaintext = b'thisisaplaintextwithspecialcharactersxy'

    def test1generate_random_key_allowed_length(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for c in key:
            self.assertTrue(c in self.ALPHABET)

        length = 19
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for c in key:
            self.assertTrue(c in self.ALPHABET)

        length = 150
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for c in key:
            self.assertTrue(c in self.ALPHABET)

    def test2generate_random_key_wrong_length_parameter(self):
        self.assertRaises(ValueError, self.cipher.generate_random_key, 0)
        self.assertRaises(ValueError, self.cipher.generate_random_key, -1)
        self.assertRaises(ValueError, self.cipher.generate_random_key, 1.55)
        self.assertRaises(ValueError, self.cipher.generate_random_key, None)

    def test3filter_keep_unknown_symbols(self):
        self.assertEqual(self.cipher.filter(self.plaintext, keep_unknown_symbols=True), self.plaintext.lower())

    def test4filter_delete_unknown_symbols(self):
        self.assertEqual(self.cipher.filter(self.plaintext, keep_unknown_symbols=False), self.filtered_plaintext)
