from cipherImplementations.monomeDinome import MonomeDinome
from util.textUtils import map_text_into_numberspace, map_numbers_into_textspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class MonomeDinomeTest(CipherTestBase):
    cipher = MonomeDinome(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'highfrequencykeysshortenciphertext'
    # ciphertext = b'6006760627539325168346553444608795168038605795359'
    # ct = []
    # for c in ciphertext:
    #     ct.append(int(bytes([c])))
    # ciphertext = map_numbers_into_textspace(ct, cipher.alphabet, cipher.unknown_symbol)
    # print(ciphertext)
    ciphertext = [60,0,67,60,62,7,5,39,32,5,1,68,34,65,5,34,4,4,60,8,7,9,5,1,68,0,38,60,5,7,9,5,35,9]
    decrypted_plaintext = b'highfrequencykeysshortenciphertext'
    key = [[6,3,1,8,9,2,7,0,5,4], map_text_into_numberspace(b'notariesbcdfghklmpquvwxy', cipher.alphabet, cipher.unknown_symbol_number)]

    def test1generate_random_key(self):
        old_key = self.cipher.alphabet
        old_numbers = self.key
        for _ in range(0, 100):
            numbers, key = self.cipher.generate_random_key()
            self.assertEqual(26, len(key))
            self.assertNotEqual(key, old_key)
            self.assertEqual(10, len(numbers))
            self.assertNotEqual(numbers, old_numbers)
            old_key = key
            old_numbers = numbers

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=False)
        plaintext_numbers = map_text_into_numberspace(plaintext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key)
        cntr = 0
        for ct in self.ciphertext:
            row = int(ct / 10)
            if row > 0:
                self.assertEqual(row, ciphertext_numbers[cntr] % 10)
                cntr += 1
            column = ct % 10
            self.assertEqual(column, ciphertext_numbers[cntr] % 10)
            cntr += 1

    def test6decrypt(self):
        ciphertext = b'gaaghgagchfdjdcfbgidegffdeeegaihjfbgiadigafhjfdfj'
        ciphertext_numbers = map_text_into_numberspace(ciphertext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key)
        plaintext = map_numbers_into_textspace(plaintext_numbers, self.cipher.alphabet, self.UNKNOWN_SYMBOL)
        self.assertEqual(self.decrypted_plaintext, plaintext)