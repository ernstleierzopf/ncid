from cipherImplementations.plaintext import Plaintext
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class PlaintextTest(CipherTestBase):
    cipher = Plaintext(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'This is some plaintext.'
    ciphertext = b'thisissomeplaintext'
    decrypted_plaintext = b'thisissomeplaintext'
    key = None

    def test1generate_random_key(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertIsNone(key)

        length = 19
        key = self.cipher.generate_random_key(length)
        self.assertIsNone(key)

        length = 120
        key = self.cipher.generate_random_key(length)
        self.assertIsNone(key)

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()