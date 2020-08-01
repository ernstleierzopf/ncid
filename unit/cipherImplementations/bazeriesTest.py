from cipherImplementations.bazeries import Bazeries
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace


class BazeriesTest(CipherTestBase):
    cipher = Bazeries(CipherTestBase.ALPHABET.replace(b'j', b''), CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'simple substitution plus transposition'
    ciphertext = b'acyyuxymrqkxkckgcrqiyitnkyxkcygqgci'
    decrypted_plaintext = b'simplesubstitutionplustransposition'
    key = [map_text_into_numberspace(b'threousandvfiywbcgklmpqxz', cipher.alphabet, cipher.unknown_symbol_number), 3752]

    def test1generate_random_key_allowed_length(self):
        for _ in range(0, 100):
            key = self.cipher.generate_random_key()
            self.assertTrue(0 <= key[1] <= 1000000)

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()