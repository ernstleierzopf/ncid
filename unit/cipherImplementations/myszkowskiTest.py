from cipherImplementations.myszkowski import Myszkowski
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class MyszkowskiTest(CipherTestBase):
    cipher = Myszkowski(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Incomplete columnar with pattern word key and letters under same number taken off by row from top to bottom.'
    key = [1,0,2,0,2,0]
    ciphertext = b'nopeeounrihatrwrkynltesnesmnmetknfbrwrmotbtoillwtoatderootocmtcmatpendederuraubaefyfopotm'
    decrypted_plaintext = b'incompletecolumnarwithpatternwordkeyandlettersundersamenumbertakenoffbyrowfromtoptobottom'

    def test1generate_random_key_allowed_length(self):
        length = 5
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        alph = list(range(int(length / 2) + 2))
        self.assertTrue(max(alph) >= max(key))

        length = 19
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        alph = list(range(int(length / 2) + 2))
        self.assertTrue(max(alph) >= max(key))

        length = 150
        key = self.cipher.generate_random_key(length)
        self.assertEqual(len(key), length)
        alph = list(range(int(length / 2) + 1))
        self.assertTrue(max(alph) >= max(key))

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.assertEqual(self.cipher.filter(self.plaintext, keep_unknown_symbols=False), self.decrypted_plaintext)

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()