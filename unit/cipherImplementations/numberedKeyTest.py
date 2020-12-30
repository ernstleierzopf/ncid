from cipherImplementations.numberedKey import NumberedKey
from cipherImplementations.cipher import OUTPUT_ALPHABET
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace, map_numbers_into_textspace


class NihilistTranspositionTest(CipherTestBase):
    cipher = NumberedKey(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'The road to success is always under construction.'
    ciphertext = b'0419202102232504022205161615222211222312072309220501252021160201220421051604170201'
    decrypted_plaintext = b'theroadtosuccessisalwaysunderconstruction'
    key = map_text_into_numberspace(b'mnoqtuvwxyzilikeciphersabdfg', cipher.alphabet, cipher.unknown_symbol_number)

    def test1generate_random_key_allowed_length(self):
        for i in range(1, 100):
            key = self.cipher.generate_random_key(i)
            self.assertTrue(len(key) >= len(self.cipher.alphabet))

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=False)
        plaintext_numbers = map_text_into_numberspace(plaintext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
        ciphertext = map_numbers_into_textspace(ciphertext_numbers, OUTPUT_ALPHABET, self.UNKNOWN_SYMBOL)
        self.assertEqual(len(ciphertext), 2 * len(plaintext))
        for i in range(0, len(ciphertext), 2):
            pos = int(bytes([ciphertext[i]])) * 10 + int(bytes([ciphertext[i + 1]]))
            self.assertEqual(self.cipher.alphabet.index(bytes([self.decrypted_plaintext[int(i / 2)]])), self.key[pos])

    def test6decrypt(self):
        self.run_test6decrypt()
