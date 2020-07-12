from cipherImplementations.simpleSubstitution import SimpleSubstitution
from util.textUtils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class SimpleSubstitutionTest(CipherTestBase):
    cipher = SimpleSubstitution(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'this is a plaintext with special characters!%%xy<'
    ciphertext = b'xokqkqfzifkuxanxwkxoqzagkfigofpfgxapqns'
    decrypted_plaintext = b'thisisaplaintextwithspecialcharactersxy'
    key = map_text_into_numberspace(b'fcghartokldibuezjpqxyvwnsm', CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)

    def test1generate_random_key(self):
        old_key = self.cipher.alphabet
        for _ in range(0, 100):
            key = self.cipher.generate_random_key()
            self.assertEqual(26, len(key))
            self.assertNotEqual(key, old_key)
            old_key = key

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()