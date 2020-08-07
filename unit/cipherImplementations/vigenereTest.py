from cipherImplementations.vigenere import Vigenere
from util.textUtils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class VigenereTest(CipherTestBase):
    cipher = Vigenere(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'In the Vigenere, C equals K plus P where A is zero, B is one, etc'
    ciphertext = b'xbefegxneoikmetefyldzwlvwiejtfpyidolrpfbaqcsprc'
    decrypted_plaintext = b'inthevigenerecequalskpluspwhereaiszerobisoneetc'
    key = map_text_into_numberspace(b'polyalphabetic', CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)

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