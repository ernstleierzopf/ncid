from cipherImplementations.grandpre import Grandpre
from util.textUtils import map_text_into_numberspace, map_numbers_into_textspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class GrandpreTest(CipherTestBase):
    cipher = Grandpre(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'The first column is the keyword.'
    # ciphertext = b'8427823456717726445464637852666584278236618873547113'
    # ct = []
    # import random
    # for c in ciphertext:
    #     c = int(bytes([c])) - 1
    #     upper = 1
    #     if c < 6:
    #         upper += 1
    #     rand = (random.randint(0, upper))
    #     ct.append(rand * 10 + c)
    # ciphertext = map_numbers_into_textspace(ct, cipher.alphabet, cipher.unknown_symbol)
    ciphertext = b'hxvghvmnezqkgqlfddynfdzwgrylzzfohdvghvcfpuhrgwyngukc'
    decrypted_plaintext = b'thefirstcolumnisthekeyword'
    key = map_text_into_numberspace(b'ladybugsazimuthscalfskinquackishunjovialevulsionrowdyismsextuply', CipherTestBase.ALPHABET,
                                    CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    key_dict = {}
    for k in set(key):
        key_dict[k] = []
    for pos, k in enumerate(key):
        row = int(pos / 8)
        column = pos % 8
        key_dict[k].append((row, column))

    def test1generate_random_key(self):
        old_key = self.cipher.alphabet
        for _ in range(0, 100):
            key = self.cipher.generate_random_key()
            self.assertEqual(len(key),26)
            self.assertNotEqual(key, old_key)
            old_key = key

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        plaintext = self.cipher.filter(self.plaintext, keep_unknown_symbols=False)
        plaintext_numbers = map_text_into_numberspace(plaintext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        ciphertext_numbers = self.cipher.encrypt(plaintext_numbers, self.key_dict)
        for i in range(0, len(plaintext_numbers), 1):
            row = ciphertext_numbers[i*2] % 10
            column = ciphertext_numbers[i*2+1] % 10
            self.assertIn((row, column), self.key_dict[plaintext_numbers[i]])

    def test6decrypt(self):
        ciphertext_numbers = map_text_into_numberspace(self.ciphertext, self.cipher.alphabet, self.UNKNOWN_SYMBOL_NUMBER)
        plaintext_numbers = self.cipher.decrypt(ciphertext_numbers, self.key_dict)
        plaintext = map_numbers_into_textspace(plaintext_numbers, self.cipher.alphabet, self.UNKNOWN_SYMBOL)
        self.assertEqual(self.decrypted_plaintext, plaintext)