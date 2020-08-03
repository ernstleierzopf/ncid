from cipherImplementations.null import Null
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace, map_numbers_into_textspace


class NihilistTranspositionTest(CipherTestBase):
    cipher = Null(CipherTestBase.ALPHABET + b' ', CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'help'
    ciphertext = b'the great old pumpers'
    decrypted_plaintext = b'help'
    key = None

    def test1generate_random_key_allowed_length(self):
        for i in range(0, 100):
            key = self.cipher.generate_random_key(i)
            self.assertIsNone(key)

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=False)
        plaintext_numbers = map_text_into_numberspace(plaintext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
        ciphertext = map_numbers_into_textspace(ciphertext_numbers, self.cipher.alphabet, self.UNKNOWN_SYMBOL)
        words = ciphertext.split(b' ')
        for i, p in enumerate(plaintext):
            self.assertEqual(words[i][int(len(words[i]) / 2)], p)

    def test6decrypt(self):
        self.run_test6decrypt()