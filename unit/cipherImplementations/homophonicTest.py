from cipherImplementations.homophonic import Homophonic
from util.textUtils import map_text_into_numberspace, map_numbers_into_textspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase
from cipherImplementations.cipher import OUTPUT_ALPHABET


class HomophonicTest(CipherTestBase):
    cipher = Homophonic(CipherTestBase.ALPHABET.replace(b'j', b''), CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'word divisions may be kept'
    ciphertext = [b'1369', b'642', b'0258', b'864', b'1258', b'197', b'2469', b'319', b'2469', b'319', b'047', b'3649', b'1369', b'531',
                  b'047', b'3649', b'1358', b'208', b'047', b'3649', b'0258', b'864', b'058', b'703', b'1358', b'208', b'0458', b'692',
                  b'2369', b'086', b'1369', b'864', b'2369', b'197', b'2470', b'420', b'0478', b'4750', b'2470', b'420', b'0258', b'975',
                  b'1358', b'319']
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
        ciphertext = map_numbers_into_textspace(ciphertext_numbers, OUTPUT_ALPHABET, self.UNKNOWN_SYMBOL)
        for i, c in enumerate(ciphertext):
            c = bytes([c])
            self.assertIn(c, self.ciphertext[i])

    def test6decrypt(self):
        ct = b'16261199694633038879548312063894672404002789'
        ciphertext_numbers = map_text_into_numberspace(ct, OUTPUT_ALPHABET, self.UNKNOWN_SYMBOL_NUMBER)
        plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key)
        plaintext = map_numbers_into_textspace(plaintext_numbers, self.cipher.alphabet, self.UNKNOWN_SYMBOL)
        self.assertEqual(self.decrypted_plaintext, plaintext)