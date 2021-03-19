from cipherImplementations.twoSquare import TwoSquare
from util.utils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class TwoSquareTest(CipherTestBase):
    cipher = TwoSquare(CipherTestBase.ALPHABET.replace(b'j', b''), CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'another digraphic setup'
    ciphertext = b'irrtehmkgimeqgrunmmzsv'
    decrypted_plaintext = b'anotherdigraphicsetupx'
    key = [map_text_into_numberspace(b'dialoguebcfhkmnpqrstvwxyz', cipher.alphabet, CipherTestBase.UNKNOWN_SYMBOL_NUMBER),
           map_text_into_numberspace(b'biographycdefklmnqstuvwxz', cipher.alphabet, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)]

    def test1generate_random_key(self):
        length = 5
        key1, key2 = self.cipher.generate_random_key(length)
        self.assertEqual(len(key1), len(self.cipher.alphabet))
        self.assertEqual(len(key2), len(self.cipher.alphabet))
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

        length = 19
        key1, key2 = self.cipher.generate_random_key(length)
        self.assertEqual(len(key1), len(self.cipher.alphabet))
        self.assertEqual(len(key2), len(self.cipher.alphabet))
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

        length = 25
        key1, key2 = self.cipher.generate_random_key(length)
        self.assertEqual(len(key1), len(self.cipher.alphabet))
        self.assertEqual(len(key2), len(self.cipher.alphabet))
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

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.assertEqual(self.cipher.filter(self.plaintext, keep_unknown_symbols=False), self.decrypted_plaintext.replace(b'x', b''))

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()
