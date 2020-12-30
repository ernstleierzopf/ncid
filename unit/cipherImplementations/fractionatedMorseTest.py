from cipherImplementations.fractionatedMorse import FractionatedMorse
from util.textUtils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class FractionatedMorseTest(CipherTestBase):
    cipher = FractionatedMorse(CipherTestBase.ALPHABET + b' ', CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'come at once.'
    ciphertext = b'cbiiltmhvvfl'
    decrypted_plaintext = b'come at once'
    key = map_text_into_numberspace(b'roundtablecfghijkmpqsvwxyz', CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)

    def test1generate_random_key(self):
        old_key = self.cipher.alphabet
        for i in range(1, 25):
            key = self.cipher.generate_random_key(i)
            self.assertEqual(26, len(key))
            self.assertNotEqual(key, old_key)

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
