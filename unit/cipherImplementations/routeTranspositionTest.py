from cipherImplementations.routeTransposition import RouteTransposition
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class RouteTranspositionTest(CipherTestBase):
    cipher = RouteTransposition(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'there are many routes'
    ciphertext = b'tharyrestuamreeeno'
    decrypted_plaintext = b'therearemanyroutes'
    key = 3

    def test1generate_random_key(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertEqual(key, length)

        length = 19
        key = self.cipher.generate_random_key(length)
        self.assertEqual(key, length)

        length = 25
        key = self.cipher.generate_random_key(length)
        self.assertEqual(key, length)

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    # def test6decrypt(self):
    #     self.run_test6decrypt()