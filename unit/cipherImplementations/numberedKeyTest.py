from cipherImplementations.numberedKey import NumberedKey
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from util.textUtils import map_text_into_numberspace, map_numbers_into_textspace


class NihilistTranspositionTest(CipherTestBase):
    cipher = NumberedKey(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'The road to success is always under construction.'
    # ct = [4,19,20,21,2,23,25,4,2,22,5,16,16,15,22,22,11,22,23,12,7,23,9,22,5,1,25,20,21,16,2,1,22,4,21,5,16,4,17,2,1]
    # ciphertext = b''
    # for c in ct:
    #     ciphertext += bytes([CipherTestBase.ALPHABET[int(c / 10)]])
    #     ciphertext += bytes([CipherTestBase.ALPHABET[int(c % 10)]])
    # print(ciphertext)
    ciphertext = b'aebjcacbaccdcfaeacccafbgbgbfccccbbcccdbcahcdajccafabcfcacbbgacabccaecbafbgaebhacab'
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
        ciphertext = map_numbers_into_textspace(ciphertext_numbers, self.cipher.alphabet, self.UNKNOWN_SYMBOL)
        self.assertEqual(len(ciphertext), 2 * len(plaintext))
        for i in range(0, len(ciphertext), 2):
            pos = self.cipher.alphabet.index(bytes([ciphertext[i]])) * 10 + self.cipher.alphabet.index(bytes([ciphertext[i + 1]]))
            self.assertEqual(self.cipher.alphabet.index(bytes([self.decrypted_plaintext[int(i / 2)]])), self.key[pos])

    def test6decrypt(self):
        self.run_test6decrypt()