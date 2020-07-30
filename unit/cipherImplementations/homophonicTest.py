from cipherImplementations.homophonic import Homophonic
from util.textUtils import map_text_into_numberspace, map_numbers_into_textspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class HomophonicTest(CipherTestBase):
    cipher = Homophonic(CipherTestBase.ALPHABET.replace(b'j', b''), CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'word divisions may be kept'
    ciphertext = [b'bdgj', b'fdb', b'acfi', b'hfd', b'bcfi', b'aig', b'cegj', b'cai', b'cegj', b'cai', b'aeh', b'cfdi', b'bdgj', b'eca',
                  b'aeh', b'cfdi', b'bcfi', b'bjh', b'aeh', b'cfdi', b'acfi', b'hfd', b'aefi', b'gjc', b'bcfi', b'bjh', b'aefi', b'fib',
                  b'bdgj', b'jhf', b'bdgj', b'hfd', b'cdgj', b'aig', b'cegj', b'dbj', b'aeh', b'dgej', b'cegj', b'dbj', b'acfi', b'ige',
                  b'bdfi', b'cai']
    # ct = [16, 26, 11, 99, 69, 46, 33, 3, 88, 79, 54, 83, 12, 6, 38, 94, 67, 24, 4, 0, 27, 89]
    # ciphertext = b''
    # for c in ct:
    #     c -= 1
    #     ciphertext += bytes([CipherTestBase.ALPHABET[int(c / 10)]])
    #     ciphertext += bytes([CipherTestBase.ALPHABET[int(c % 10)]])
    # print(ciphertext)
    decrypted_plaintext = b'worddivisionsmaybekept'
    key = map_text_into_numberspace(b'golf', cipher.alphabet, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)

    def test1generate_random_key(self):
        old_key = self.cipher.alphabet
        for _ in range(0, 100):
            key = self.cipher.generate_random_key()
            self.assertEqual(4, len(key))
            self.assertNotEqual(key, old_key)
            old_key = key

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=False)
        plaintext_numbers = map_text_into_numberspace(plaintext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
        ciphertext = map_numbers_into_textspace(ciphertext_numbers, CipherTestBase.ALPHABET, self.UNKNOWN_SYMBOL)
        for i, c in enumerate(ciphertext):
            c = bytes([c])
            self.assertIn(c, self.ciphertext[i])

    def test6decrypt(self):
        ct = b'bfahbaccccacbeacbbacahagbbafbkbhcacdadcdaibc'
        ciphertext_numbers = map_text_into_numberspace(ct, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key)
        plaintext = map_numbers_into_textspace(plaintext_numbers, self.cipher.alphabet, self.UNKNOWN_SYMBOL)
        self.assertEqual(self.decrypted_plaintext, plaintext)