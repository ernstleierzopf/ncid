from cipherImplementations.nicodemus import Nicodemus
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace


class NicodemusTest(CipherTestBase):
    cipher = Nicodemus(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'the early bird gets the worm'
    key = map_text_into_numberspace(b'cat', cipher.alphabet, cipher.unknown_symbol_number)
    ciphertext = b'hayrevgnkixkuwmtwmugtah'
    decrypted_plaintext = b'theearlybirdgetstheworm'

    def test1generate_random_key_allowed_length(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for c in key:
            self.assertIn(c, self.cipher.alphabet)

        length = 19
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        for c in key:
            self.assertIn(c, self.cipher.alphabet)

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.assertEqual(self.cipher.filter(self.plaintext, keep_unknown_symbols=False), self.decrypted_plaintext.replace(b'x', b''))

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()
