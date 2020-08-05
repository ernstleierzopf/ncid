from cipherImplementations.quagmire import Quagmire
from util.textUtils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class Quagmire1Test(CipherTestBase):
    cipher = Quagmire(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER, keyword_type=1)
    plaintext = b'The Quag One is a periodic cipher with a keyed plain alphabet run against a straight cipher alphabet.'
    ciphertext = b'qpmgqrbujuyifdmpyaifqyyjjjhjycjluutpidvwymfsgaesdwhizrblirvcfczpelbpzyyjjjhwljjlpup'
    decrypted_plaintext = b'thequagoneisaperiodiccipherwithakeyedplainalphabetrunagainstastraightcipheralphabet'
    key = [map_text_into_numberspace(b'flower', cipher.alphabet, CipherTestBase.UNKNOWN_SYMBOL_NUMBER),
           map_text_into_numberspace(b'springfevabcdhjklmoqtuwxyz', cipher.alphabet, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)]

    def test1generate_random_key(self):
        length = 5
        keyword, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        self.assertEqual(len(keyword), length)
        alphabet2 = b'' + self.cipher.alphabet
        for c in key:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')

        length = 19
        keyword, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        self.assertEqual(len(keyword), length)
        alphabet2 = b'' + self.cipher.alphabet
        for c in key:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')

        length = 25
        keyword, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        self.assertEqual(len(keyword), length)
        alphabet2 = b'' + self.cipher.alphabet
        for c in key:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')

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