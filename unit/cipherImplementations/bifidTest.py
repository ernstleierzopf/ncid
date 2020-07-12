from cipherImplementations.bifid import Bifid
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class BifidTest(CipherTestBase):
    cipher = Bifid(CipherTestBase.ALPHABET.replace(b'j', b''), CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Odd periods are popular.'
    ciphertext = b'mweingimgeoyyrlveywy'
    decrypted_plaintext = b'oddperiodsarepopular'
    key = [7, b'extraklmpohwzqdgvusifcbyn']

    def test1generate_random_key_allowed_length(self):
        length = 5
        leng, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        self.assertEqual(leng, length)
        for c in key:
            self.assertTrue(c in self.ALPHABET)

        length = 19
        leng, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        self.assertEqual(leng, length)
        for c in key:
            self.assertTrue(c in self.ALPHABET)

        length = 150
        leng, key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), len(self.cipher.alphabet))
        self.assertEqual(leng, length)
        for c in key:
            self.assertTrue(c in self.ALPHABET)

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