from cipherImplementations.gronsfeld import Gronsfeld
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class GronsfeldTest(CipherTestBase):
    cipher = Gronsfeld(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'This one uses ten of the twenty-six Vigenere alphabets.'
    ciphertext = b'ckktswgdvgtxnpxiviicynqvzwrzelifrntndnqljdnwu'
    decrypted_plaintext = b'thisoneusestenofthetwentysixvigenerealphabets'
    key = [9,3,2,1,4,9,2]

    def test1generate_random_key(self):
        for i in range(1, 100):
            key = self.cipher.generate_random_key(i)
            self.assertEqual(i, len(key))

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