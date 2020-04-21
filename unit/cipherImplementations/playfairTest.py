import unittest
from cipherImplementations.playfair import Playfair
import unit.cipherImplementations.cipherTest as cipherTest
import util.textUtils as text_utils


class PlayfairTest(unittest.TestCase):
    CipherTest = cipherTest.CipherTest
    cipher = Playfair(CipherTest.ALPHABET.replace(b'j', b''), b'x', ord('x'))
    plaintext = b'this is a plaintext with special characters!%xzll'
    #ciphertext_keep_unknown_symbols = b'xdnotnxyytqkdoixhuuttmxdyxukalcivydyozavgpxyyhwynvnv'
    ciphertext_remove_unknown_symbols = b'xdnonoboickiudtutmxdoqfbodqfdyozavgpxycmvn'
    #decrypted_plaintext_keep_unknown_symbols = b'thisxisxaxplaintextxwithxspecialxcharactersxxxzlxl'
    decrypted_plaintext_remove_unknown_symbols = b'thisisaplaintextwithspecialcharactersxzlxl'
    key = text_utils.map_text_into_numberspace(b'abczydefghiklmnopqrstuvwx', cipher.alphabet, cipher.unknown_symbol_number)

    def test1generate_random_key_allowed_length(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        for c in key:
            self.assertTrue(c in self.cipher.alphabet)

        length = 19
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        for c in key:
            self.assertTrue(c in self.cipher.alphabet)

        length = 25
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        for c in key:
            self.assertTrue(c in self.cipher.alphabet)

    def test2generate_random_key_wrong_length_parameter(self):
        self.assertRaises(ValueError, self.cipher.generate_random_key, 0)
        self.assertRaises(ValueError, self.cipher.generate_random_key, -1)
        self.assertRaises(ValueError, self.cipher.generate_random_key, 1.55)
        self.assertRaises(ValueError, self.cipher.generate_random_key, None)
        self.assertRaises(ValueError, self.cipher.generate_random_key, 27)

    # def test3encrypt_keep_unknown_symbols(self):
    #     plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=True)
    #     plaintext_numbers = text_utils.map_text_into_numberspace(plaintext, self.cipher.alphabet, self.cipher.unknown_symbol_number)
    #     ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
    #     ciphertext = text_utils.map_numbers_into_textspace(ciphertext_numbers, self.cipher.alphabet, self.cipher.unknown_symbol)
    #     self.assertEqual(self.ciphertext_keep_unknown_symbols, ciphertext)

    def test4encrypt_remove_unknown_symbols(self):
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=False)
        plaintext_numbers = text_utils.map_text_into_numberspace(plaintext, self.cipher.alphabet, self.cipher.unknown_symbol_number)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
        ciphertext = text_utils.map_numbers_into_textspace(ciphertext_numbers, self.cipher.alphabet, self.cipher.unknown_symbol)
        self.assertEqual(self.ciphertext_remove_unknown_symbols, ciphertext)

    # def test5decrypt_keep_unknown_symbols(self):
    #     ciphertext_numbers = text_utils.map_text_into_numberspace(self.ciphertext_keep_unknown_symbols, self.cipher.alphabet, self.cipher.unknown_symbol_number)
    #     plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key)
    #     plaintext = text_utils.map_numbers_into_textspace(plaintext_numbers, self.cipher.alphabet, self.cipher.unknown_symbol)
    #     self.assertEqual(self.decrypted_plaintext_keep_unknown_symbols, plaintext)

    def test6decrypt_remove_unknown_symbols(self):
        ciphertext_numbers = text_utils.map_text_into_numberspace(self.ciphertext_remove_unknown_symbols, self.cipher.alphabet, self.cipher.unknown_symbol_number)
        plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key)
        plaintext = text_utils.map_numbers_into_textspace(plaintext_numbers, self.cipher.alphabet, self.cipher.unknown_symbol)
        self.assertEqual(self.decrypted_plaintext_remove_unknown_symbols, plaintext)
