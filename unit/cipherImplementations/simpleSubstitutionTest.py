import unittest
from cipherImplementations.simpleSubstitution import SimpleSubstitution
import unit.cipherImplementations.cipherTest as cipherTest
import util.textUtils as text_utils


class SimpleSubstitutionTest(unittest.TestCase):
    CipherTest = cipherTest.CipherTest
    cipher = SimpleSubstitution(CipherTest.ALPHABET, CipherTest.UNKNOWN_SYMBOL, CipherTest.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'this is a plaintext with special characters!%%xy<'
    ciphertext_keep_unknown_symbols = b'xokq?kq?f?zifkuxanx?wkxo?qzagkfi?gofpfgxapq???ns?'
    ciphertext_remove_unknown_symbols = b'xokqkqfzifkuxanxwkxoqzagkfigofpfgxapqns'
    decrypted_plaintext_keep_unknown_symbols = b'this?is?a?plaintext?with?special?characters???xy?'
    decrypted_plaintext_remove_unknown_symbols = b'thisisaplaintextwithspecialcharactersxy'
    key = text_utils.map_text_into_numberspace(b'fcghartokldibuezjpqxyvwnsm', CipherTest.ALPHABET, CipherTest.UNKNOWN_SYMBOL_NUMBER)

    def test1generate_random_key(self):
        old_key = self.cipher.alphabet
        for _ in range(0, 100):
            key = self.cipher.generate_random_key()
            self.assertEqual(26, len(key))
            self.assertNotEqual(key, old_key)
            old_key = key

    def test2encrypt_keep_unknown_symbols(self):
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=True)
        plaintext_numbers = text_utils.map_text_into_numberspace(plaintext, self.CipherTest.ALPHABET, self.CipherTest.UNKNOWN_SYMBOL_NUMBER)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
        ciphertext = text_utils.map_numbers_into_textspace(ciphertext_numbers, self.CipherTest.ALPHABET, self.CipherTest.UNKNOWN_SYMBOL)
        self.assertEqual(self.ciphertext_keep_unknown_symbols, ciphertext)

    def test3encrypt_remove_unknown_symbols(self):
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=False)
        plaintext_numbers = text_utils.map_text_into_numberspace(plaintext, self.CipherTest.ALPHABET, self.CipherTest.UNKNOWN_SYMBOL_NUMBER)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
        ciphertext = text_utils.map_numbers_into_textspace(ciphertext_numbers, self.CipherTest.ALPHABET, self.CipherTest.UNKNOWN_SYMBOL)
        self.assertEqual(self.ciphertext_remove_unknown_symbols, ciphertext)

    def test4decrypt_keep_unknown_symbols(self):
        ciphertext_numbers = text_utils.map_text_into_numberspace(self.ciphertext_keep_unknown_symbols, self.CipherTest.ALPHABET, self.CipherTest.UNKNOWN_SYMBOL_NUMBER)
        plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key)
        plaintext = text_utils.map_numbers_into_textspace(plaintext_numbers, self.CipherTest.ALPHABET, self.CipherTest.UNKNOWN_SYMBOL)
        self.assertEqual(self.decrypted_plaintext_keep_unknown_symbols, plaintext)

    def test4decrypt_remove_unknown_symbols(self):
        ciphertext_numbers = text_utils.map_text_into_numberspace(self.ciphertext_remove_unknown_symbols, self.CipherTest.ALPHABET, self.CipherTest.UNKNOWN_SYMBOL_NUMBER)
        plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key)
        plaintext = text_utils.map_numbers_into_textspace(plaintext_numbers, self.CipherTest.ALPHABET, self.CipherTest.UNKNOWN_SYMBOL)
        self.assertEqual(self.decrypted_plaintext_remove_unknown_symbols, plaintext)
