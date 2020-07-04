from unit.cipherImplementations.CipherTestBase import CipherTestBase
from cipherImplementations.cipher import Cipher


class CipherTest(CipherTestBase):
    cipher = Cipher()
    cipher.alphabet = CipherTestBase.ALPHABET
    plaintext = b'This is a plaintext with special characters!%%xy<'
    decrypted_plaintext = b'thisisaplaintextwithspecialcharactersxy'

    def test1generate_random_key_allowed_length(self):
        self.run_test1generate_random_key_allowed_length()

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()