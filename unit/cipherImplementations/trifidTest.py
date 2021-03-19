from cipherImplementations.trifid import Trifid
from util.utils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class TrifidTest(CipherTestBase):
    cipher = Trifid(CipherTestBase.ALPHABET + b'#', CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'trifids are fractionated ciphers'
    ciphertext = b'eymxvucryyyyeayvyovvxitdpathe'
    decrypted_plaintext = b'trifidsarefractionatedciphers'
    key = [map_text_into_numberspace(b'extraodinybcfghjklmpqsuvwz#', cipher.alphabet, CipherTestBase.UNKNOWN_SYMBOL_NUMBER), 10]

    def test1generate_random_key(self):
        length = 5
        key, leng = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        alphabet2 = CipherTestBase.ALPHABET + b'#'
        for c in key:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')
        self.assertEqual(leng, length)

        length = 19
        key, leng = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        alphabet2 = CipherTestBase.ALPHABET + b'#'
        for c in key:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')
        self.assertEqual(leng, length)

        length = 25
        key, leng = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        alphabet2 = CipherTestBase.ALPHABET + b'#'
        for c in key:
            self.assertIn(c, alphabet2)
            alphabet2 = alphabet2.replace(bytes([c]), b'')
        self.assertEqual(alphabet2, b'')
        self.assertEqual(leng, length)

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
