from cipherImplementations.baconian import Baconian
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace


class BaconianTest(CipherTestBase):
    cipher = Baconian(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'now is a good t'
    ciphertext = b'bowedasterpinedjokedtheirblackhastearrayinsetchestsling'
    decrypted_plaintext = b'nowisagoodt'
    key = cipher.generate_random_key()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=False)
        plaintext_numbers = map_text_into_numberspace(plaintext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
        self.assertEqual(len(plaintext_numbers) * 5, len(ciphertext_numbers))

    def test6decrypt(self):
        self.run_test6decrypt()