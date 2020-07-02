import unittest
from cipherImplementations.hill import Hill
import unit.cipherImplementations.cipherTest as cipherTest
import util.textUtils as text_utils


class HillTest(unittest.TestCase):
    CipherTest = cipherTest.CipherTest
    UNKNOWN_SYMBOL = b'x'
    UNKNOWN_SYMBOL_NUMBER = ord('x')
    cipher = Hill(CipherTest.ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'this is a plaintext with special characters!%%xy<d'
    ciphertext_keep_unknown_symbols = b'jufdhzbegqjevbslimxofpeniofrbybswwfxfnniajubhvaaejpu'
    ciphertext_remove_unknown_symbols = b'jufdtmdkdtheluizfpenrzzherhtyyvbmropizoz'
    decrypted_plaintext_keep_unknown_symbols = b'this?is?a?plaintext?with?special?characters???xy?d'
    decrypted_plaintext_remove_unknown_symbols = b'thisisaplaintextwithspecialcharactersxyd'
    key = [[2,15,22,3], [1,9,1,12], [16,7,13,11], [8,5,9,6]]

    def test1generate_random_key(self):
        for _ in range(0, 10):
            key = self.cipher.generate_random_key()
            self.assertEqual(4, len(key))
            for arr in key:
                self.assertEqual(4, len(arr))

    # def test2encrypt_keep_unknown_symbols(self):
    #     plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=True)
    #     plaintext_numbers = text_utils.map_text_into_numberspace(plaintext, self.CipherTest.ALPHABET, self.UNKNOWN_SYMBOL_NUMBER)
    #     ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
    #     ciphertext = text_utils.map_numbers_into_textspace(ciphertext_numbers, self.CipherTest.ALPHABET, self.UNKNOWN_SYMBOL)
    #     self.assertEqual(self.ciphertext_keep_unknown_symbols, ciphertext)

    def test3encrypt_remove_unknown_symbols(self):
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=False)
        plaintext_numbers = text_utils.map_text_into_numberspace(plaintext, self.CipherTest.ALPHABET, self.UNKNOWN_SYMBOL_NUMBER)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
        ciphertext = text_utils.map_numbers_into_textspace(ciphertext_numbers, self.CipherTest.ALPHABET, self.UNKNOWN_SYMBOL)
        self.assertEqual(self.ciphertext_remove_unknown_symbols, ciphertext)

    # def test4decrypt_keep_unknown_symbols(self):
    #     ciphertext_numbers = text_utils.map_text_into_numberspace(self.ciphertext_keep_unknown_symbols, self.CipherTest.ALPHABET, self.UNKNOWN_SYMBOL_NUMBER)
    #     plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key)
    #     plaintext = text_utils.map_numbers_into_textspace(plaintext_numbers, self.CipherTest.ALPHABET, self.UNKNOWN_SYMBOL)
    #     self.assertEqual(self.decrypted_plaintext_keep_unknown_symbols, plaintext)

    # def test5decrypt_remove_unknown_symbols(self):
    #     ciphertext_numbers = text_utils.map_text_into_numberspace(self.ciphertext_remove_unknown_symbols, self.CipherTest.ALPHABET, self.UNKNOWN_SYMBOL_NUMBER)
    #     plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key)
    #     plaintext = text_utils.map_numbers_into_textspace(plaintext_numbers, self.CipherTest.ALPHABET, self.UNKNOWN_SYMBOL)
    #     self.assertEqual(self.decrypted_plaintext_remove_unknown_symbols, plaintext)
