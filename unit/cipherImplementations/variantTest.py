from cipherImplementations.variant import Variant
from util.utils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class VariantTest(CipherTestBase):
    cipher = Variant(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'C equals P minus K.'
    ciphertext = b'cpbjwldabenfdz'
    decrypted_plaintext = b'cequalspminusk'
    key = map_text_into_numberspace(b'apple', CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)

    def test1generate_random_key_allowed_length(self):
        self.run_test1generate_random_key_allowed_length()

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
