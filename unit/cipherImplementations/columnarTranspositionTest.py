from cipherImplementations.columnarTransposition import ColumnarTransposition
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace


class ColumnarTranspositionTest(CipherTestBase):
    cipher = ColumnarTransposition(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'this is a plaintext with special characters!%%xy<'
    ciphertext = b'tlwichaiatiitlesnhcritshssepaxaxeryptca'
    decrypted_plaintext = b'thisisaplaintextwithspecialcharactersxy'
    key = map_text_into_numberspace(b'aaabcdef', CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)

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