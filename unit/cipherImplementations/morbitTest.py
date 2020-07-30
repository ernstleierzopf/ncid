from cipherImplementations.morbit import Morbit
from util.textUtils import map_text_into_numberspace, map_numbers_into_textspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class MorbitTest(CipherTestBase):
    cipher = Morbit(CipherTestBase.ALPHABET + b' ', CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Once upon a time.'
    # ciphertext = b'27435881512827465679378'
    # ct = []
    # import random
    # for c in ciphertext:
    #     c = int(bytes([c]))
    #     upper = 1
    #     if c < 6:
    #         upper += 1
    #     rand = (random.randint(0, upper))
    #     ct.append(rand * 10 + c)
    # ciphertext = map_numbers_into_textspace(ct, cipher.alphabet, cipher.unknown_symbol)
    # print(ciphertext)
    ciphertext = [2,7,4,3,5,8,8,1,5,1,2,8,2,7,4,6,5,6,7,9,3,7,8]
    decrypted_plaintext = b'once upon a time'
    key = [9,5,8,4,2,7,1,3,6]

    def test1generate_random_key(self):
        for _ in range(0, 100):
            key = self.cipher.generate_random_key()
            self.assertEqual(9, len(key))

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
        ciphertext = b'whenpsibzbwsmhogpghtdri'
        ciphertext_numbers = map_text_into_numberspace(ciphertext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key)
        plaintext = map_numbers_into_textspace(plaintext_numbers, self.cipher.alphabet, self.UNKNOWN_SYMBOL)
        self.assertEqual(self.decrypted_plaintext, plaintext)