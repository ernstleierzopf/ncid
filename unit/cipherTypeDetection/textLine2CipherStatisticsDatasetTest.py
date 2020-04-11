import unittest
from collections import Counter
from cipherTypeDetection import textLine2CipherStatisticsDataset as textLine2CipherStatisticsDataset
import unit.cipherImplementations.cipherTest as cipherTest
import util.text_utils as text_utils
import numpy as np


class TextLine2CipherStatisticsDatasetTest(unittest.TestCase):
    cipher = cipherTest.CipherTest.cipher
    ALPHABET = text_utils.map_text_into_numberspace(cipher.alphabet, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'thisisafilteredplaintextwithsomewordsinittobeusedtotestthestatisticsofthetextlinetocipherstatisticsd'
    plaintext_numberspace = text_utils.map_text_into_numberspace(plaintext, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
    # key = fcghartokldibuezjpqxyvwnsm
    simple_substitution_ciphertext = b'xokqkqfrkixapahzifkuxanxwkxoqebawephqkukxxecayqahxexaqxxoaqxfxkqxkgqerxoaxanxikuaxegkzoapqxfxkqxkgqh'
    simple_substitution_ciphertext_numberspace = text_utils.map_text_into_numberspace(simple_substitution_ciphertext, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)

    def test01calculate_frequencies_non_recursive(self):
        # unigrams
        plaintext_counter = Counter(self.plaintext.decode())
        simple_substitution_ciphertext_counter = Counter(self.simple_substitution_ciphertext.decode())
        unigram_frequencies_plaintext = [0]*26
        unigram_frequencies_simple_substitution_ciphertext = [0]*26
        for i, c in enumerate(self.cipher.alphabet.decode()):
            unigram_frequencies_plaintext[i] = plaintext_counter[c] / len(self.plaintext)
            unigram_frequencies_simple_substitution_ciphertext[i] = simple_substitution_ciphertext_counter[c] / len(self.simple_substitution_ciphertext)

        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_frequencies(self.plaintext_numberspace, 1, recursive=False), unigram_frequencies_plaintext)
        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_frequencies(self.simple_substitution_ciphertext_numberspace, 1, recursive=False), unigram_frequencies_simple_substitution_ciphertext)

        keys_plaintext = list(plaintext_counter.keys())
        keys_simple_substitution_cipher = list(simple_substitution_ciphertext_counter.keys())
        for i in range(0, len(keys_plaintext)):
            self.assertEqual(plaintext_counter[keys_plaintext[i]], simple_substitution_ciphertext_counter[keys_simple_substitution_cipher[i]])

        # bigrams
        bigram_frequencies_simple_substitution_ciphertext = [0]*676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                cntr = 0
                for k in range(0, len(self.simple_substitution_ciphertext)-1):
                    if self.simple_substitution_ciphertext[k] == self.cipher.alphabet[i] and \
                        self.simple_substitution_ciphertext[k+1] == self.cipher.alphabet[j]:
                        cntr += 1
                bigram_frequencies_simple_substitution_ciphertext[i*26+j] = cntr / len(self.simple_substitution_ciphertext)

        bigram_frequencies_plaintext = [0]*676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                cntr = 0
                for k in range(0, len(self.plaintext)-1):
                    if self.plaintext[k] == self.cipher.alphabet[i] and \
                        self.plaintext[k+1] == self.cipher.alphabet[j]:
                        cntr += 1
                bigram_frequencies_plaintext[i*26+j] = cntr / len(self.plaintext)

        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_frequencies(self.plaintext_numberspace, 2, recursive=False), bigram_frequencies_plaintext)
        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_frequencies(self.simple_substitution_ciphertext_numberspace, 2, recursive=False), bigram_frequencies_simple_substitution_ciphertext)

        # trigrams
        trigram_frequencies_simple_substitution_ciphertext = [0]*17576
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                for k in range(0, len(self.cipher.alphabet)):
                    cntr = 0
                    for l in range(0, len(self.simple_substitution_ciphertext) - 2):
                        if self.simple_substitution_ciphertext[l] == self.cipher.alphabet[i] and \
                                self.simple_substitution_ciphertext[l + 1] == self.cipher.alphabet[j]\
                                and self.simple_substitution_ciphertext[l + 2] == self.cipher.alphabet[k]:
                            cntr += 1
                    trigram_frequencies_simple_substitution_ciphertext[i * 676 + j * 26 + k] = cntr / len(
                        self.simple_substitution_ciphertext)

        trigram_frequencies_plaintext = [0]*17576
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                for k in range(0, len(self.cipher.alphabet)):
                    cntr = 0
                    for l in range(0, len(self.plaintext) - 2):
                        if self.plaintext[l] == self.cipher.alphabet[i] and \
                                self.plaintext[l + 1] == self.cipher.alphabet[j]\
                                and self.plaintext[l + 2] == self.cipher.alphabet[k]:
                            cntr += 1
                    trigram_frequencies_plaintext[i * 676 + j * 26 + k] = cntr / len(
                        self.simple_substitution_ciphertext)

        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_frequencies(self.plaintext_numberspace, 3, recursive=False), trigram_frequencies_plaintext)
        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_frequencies(self.simple_substitution_ciphertext_numberspace, 3, recursive=False), trigram_frequencies_simple_substitution_ciphertext)


    def test02calculate_frequencies_recursive(self):
        pass

    def test03calculate_ny_gram_frequencies_non_recursive(self):
        pass

    def test04calculate_ny_gram_frequencies_recursive(self):
        pass

    def test05calculate_index_of_coincidence(self):
        pass

    def test06calculate_index_of_coincidence_bigrams(self):
        pass

    # def test07has_letter_j(self):
    #     pass
    #
    # def test08has_doubles(self):
    #     pass
    #
    # def test09calculate_chi_square(self):
    #     pass
    #
    # def test10pattern_repetitions(self):
    #     pass
    #
    # def test11calculate_entropy(self):
    #     pass
    #
    # def test12calculate_autocorrelation(self):
    #     pass

    def test13encrypt(self):
        pass

    def test14calculate_statistics(self):
        pass