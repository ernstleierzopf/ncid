from cipherImplementations.condi import Condi
from util.textUtils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase


class CondiTest(CipherTestBase):
    cipher = Condi(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'Ours is a very green pastime. The wide variety of ciphers we use can all be solved with pencil and paper.'
    ciphertext = b'morcppdnbkedjkpmrtdbqcrjpxcktvbnhuyjgvbysfdxkcrcesiuojjyfrqiximbvhkqpdnmpsbyjqbttnk'
    decrypted_plaintext = b'oursisaverygreenpastimethewidevarietyofciphersweusecanallbesolvedwithpencilandpaper'
    key = [map_text_into_numberspace(b'vwxyzstrangebcdfhijklmopqu', CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL_NUMBER), 25]

    def test1generate_random_key(self):
        old_key = self.cipher.alphabet
        for _ in range(0, 100):
            key, offset = self.cipher.generate_random_key()
            self.assertEqual(26, len(key))
            self.assertNotEqual(key, old_key)
            self.assertTrue(offset < len(key))
            old_key = key

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()