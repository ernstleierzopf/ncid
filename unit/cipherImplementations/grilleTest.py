from cipherImplementations.grille import Grille
from util.textUtils import map_text_into_numberspace
from unit.cipherImplementations.CipherTestBase import CipherTestBase
import numpy as np


class GrilleTest(CipherTestBase):
    cipher = Grille(CipherTestBase.ALPHABET, CipherTestBase.UNKNOWN_SYMBOL, CipherTestBase.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'the turning grille'
    # plaintext = b'the turning grillethe turning grillethe turning grillethe turning grille'
    ciphertext = b'tilunrghgeltenir'
    # ciphertext = b'tilunrghgeltenirtilunrghgeltenirtilunrghgeltenirtilunrghgeltenir'
    decrypted_plaintext = b'theturninggrille'
    # decrypted_plaintext = b'theturninggrilletheturninggrilletheturninggrilletheturninggrille'
    key = np.array([[1,0,0,0],
                    [0,0,0,1],
                    [0,1,0,1],
                    [0,0,0,0]])

    def test1generate_random_key(self):
        for i in range(2, 100):
            key = self.cipher.generate_random_key(i)
            self.assertEqual(len(key), i)
            count = np.count_nonzero(key)
            self.assertEqual(count, i)

    def test2generate_random_key_wrong_length_parameter(self):
        self.run_test2generate_random_key_wrong_length_parameter()
        self.assertRaises(ValueError, self.cipher.generate_random_key, 1)

    def test3filter_keep_unknown_symbols(self):
        self.run_test3filter_keep_unknown_symbols()

    def test4filter_delete_unknown_symbols(self):
        self.run_test4filter_delete_unknown_symbols()

    def test5encrypt(self):
        self.run_test5encrypt()

    def test6decrypt(self):
        self.run_test6decrypt()