from cipherImplementations.keyPhrase import KeyPhrase
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.utils import map_text_into_numberspace


class IncompleteColumnarTranspositionTest(CipherTestBase):
    cipher = KeyPhrase(CipherTestBase.ALPHABET + b' ', CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'a ciphertext letter may stand for more than one plaintext letter.'
    key = map_text_into_numberspace(b'givemelibertyorgivemedeath', cipher.alphabet, cipher.unknown_symbol_number)
    ciphertext = b'g vbgimvmmam tmmmmv ygt emgoe erv yrvm migo rom gtgbommam tmmmmv'
    decrypted_plaintext = b'a ciphertext letter may stand for more than one plaintext letter'

    def test1generate_random_key(self):
        self.run_test1generate_random_alphabet()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()
