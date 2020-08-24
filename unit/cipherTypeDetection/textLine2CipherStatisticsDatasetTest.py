import unittest
from collections import Counter
from cipherTypeDetection import textLine2CipherStatisticsDataset as ds
import unit.cipherImplementations.cipherTest as cipherTest
from util.textUtils import map_text_into_numberspace
import math
import numpy as np
from cipherImplementations.cipher import OUTPUT_ALPHABET


class TextLine2CipherStatisticsDatasetTest(unittest.TestCase):
    cipher = cipherTest.CipherTest.cipher
    ALPHABET = map_text_into_numberspace(cipher.alphabet, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'thisisafilteredplaintextwithsomewordsinittobeusedtotestthestatisticsofthetextlinetocipherstatistjcsd'
    plaintext_numberspace = map_text_into_numberspace(plaintext, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
    plaintext_numberspace = [int(d) for d in plaintext_numberspace]
    # key = fcghartokldibuezjpqxyvwnsm
    # cipher: simple substitution
    ciphertext = b'xokqkqfrkixapahzifkuxanxwkxoqebawephqkukxxecayqahxexaqxxoaqxfxkqxkgqerxoaxanxikuaxegkzoapqxfxkqxlgqh'
    ciphertext_numberspace = map_text_into_numberspace(ciphertext, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
    ciphertext_numberspace = [int(d) for d in ciphertext_numberspace]

    def test01calculate_frequencies(self):
        # unigrams
        alph_size = len(OUTPUT_ALPHABET)
        squared_alph_size = len(OUTPUT_ALPHABET) * len(OUTPUT_ALPHABET)
        third_pow_size = len(OUTPUT_ALPHABET) * len(OUTPUT_ALPHABET) * len(OUTPUT_ALPHABET)
        plaintext_counter = Counter(self.plaintext.decode())
        ciphertext_counter = Counter(self.ciphertext.decode())
        unigram_frequencies_plaintext = [0]*alph_size
        unigram_frequencies_ciphertext = [0]*alph_size
        for i, c in enumerate(OUTPUT_ALPHABET.decode()):
            unigram_frequencies_plaintext[i] = plaintext_counter[c] / len(self.plaintext)
            unigram_frequencies_ciphertext[i] = ciphertext_counter[c] / len(self.ciphertext)

        self.assertCountEqual(ds.calculate_frequencies(self.plaintext_numberspace, 1, recursive=False), unigram_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_frequencies(self.ciphertext_numberspace, 1, recursive=False), unigram_frequencies_ciphertext)

        keys_plaintext = list(plaintext_counter.keys())
        keys_simple_substitution_cipher = list(ciphertext_counter.keys())
        for i in range(0, len(keys_plaintext)):
            self.assertEqual(plaintext_counter[keys_plaintext[i]], ciphertext_counter[keys_simple_substitution_cipher[i]])

        # bigrams
        bigram_frequencies_ciphertext = [0]*squared_alph_size
        for i in range(0, len(OUTPUT_ALPHABET)):
            for j in range(0, len(OUTPUT_ALPHABET)):
                cntr = 0
                for k in range(0, len(self.ciphertext) - 1):
                    if self.ciphertext[k] == OUTPUT_ALPHABET[i] and \
                            self.ciphertext[k + 1] == OUTPUT_ALPHABET[j]:
                        cntr += 1
                bigram_frequencies_ciphertext[i*alph_size+j] = cntr / len(self.ciphertext)

        bigram_frequencies_plaintext = [0]*squared_alph_size
        for i in range(0, len(OUTPUT_ALPHABET)):
            for j in range(0, len(OUTPUT_ALPHABET)):
                cntr = 0
                for k in range(0, len(self.plaintext)-1):
                    if self.plaintext[k] == OUTPUT_ALPHABET[i] and \
                            self.plaintext[k+1] == OUTPUT_ALPHABET[j]:
                        cntr += 1
                bigram_frequencies_plaintext[i*alph_size+j] = cntr / len(self.plaintext)

        self.assertCountEqual(ds.calculate_frequencies(self.plaintext_numberspace, 2, recursive=False), bigram_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_frequencies(self.ciphertext_numberspace, 2, recursive=False), bigram_frequencies_ciphertext)

        # trigrams
        trigram_frequencies_ciphertext = [0]*third_pow_size
        for i in range(0, len(OUTPUT_ALPHABET)):
            for j in range(0, len(OUTPUT_ALPHABET)):
                for k in range(0, len(OUTPUT_ALPHABET)):
                    cntr = 0
                    for pos in range(0, len(self.ciphertext) - 2):
                        if self.ciphertext[pos] == OUTPUT_ALPHABET[i] and self.ciphertext[pos + 1] == OUTPUT_ALPHABET[j]\
                                and self.ciphertext[pos + 2] == OUTPUT_ALPHABET[k]:
                            cntr += 1
                    trigram_frequencies_ciphertext[i * squared_alph_size + j * alph_size + k] = cntr / len(self.ciphertext)

        trigram_frequencies_plaintext = [0]*third_pow_size
        for i in range(0, len(OUTPUT_ALPHABET)):
            for j in range(0, len(OUTPUT_ALPHABET)):
                for k in range(0, len(OUTPUT_ALPHABET)):
                    cntr = 0
                    for pos in range(0, len(self.plaintext) - 2):
                        if self.plaintext[pos] == OUTPUT_ALPHABET[i] and self.plaintext[pos + 1] == OUTPUT_ALPHABET[j]\
                                and self.plaintext[pos + 2] == OUTPUT_ALPHABET[k]:
                            cntr += 1
                    trigram_frequencies_plaintext[i * squared_alph_size + j * alph_size + k] = cntr / len(self.ciphertext)

        self.assertCountEqual(ds.calculate_frequencies(self.plaintext_numberspace, 3, recursive=False), trigram_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_frequencies(self.ciphertext_numberspace, 3, recursive=False), trigram_frequencies_ciphertext)

        self.assertCountEqual(ds.calculate_frequencies(self.plaintext_numberspace, 3, recursive=True),
                              unigram_frequencies_plaintext + bigram_frequencies_plaintext + trigram_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_frequencies(self.ciphertext_numberspace, 3, recursive=True),
                              unigram_frequencies_ciphertext + bigram_frequencies_ciphertext + trigram_frequencies_ciphertext)

    def test02calculate_ny_gram_frequencies(self):
        alph_size = len(OUTPUT_ALPHABET)
        squared_alph_size = len(OUTPUT_ALPHABET) * len(OUTPUT_ALPHABET)
        third_pow_size = len(OUTPUT_ALPHABET) * len(OUTPUT_ALPHABET) * len(OUTPUT_ALPHABET)
        # bigrams interval=2
        bigram_ny_frequencies_ciphertext = [0] * squared_alph_size
        for i in range(0, len(OUTPUT_ALPHABET)):
            for j in range(0, len(OUTPUT_ALPHABET)):
                cntr = 0
                for k in range(0, len(self.ciphertext) - 2):
                    if self.ciphertext[k] == OUTPUT_ALPHABET[i] and \
                            self.ciphertext[k + 2] == OUTPUT_ALPHABET[j]:
                        cntr += 1
                bigram_ny_frequencies_ciphertext[i * alph_size + j] = cntr / len(self.ciphertext)

        bigram_ny_frequencies_plaintext = [0] * squared_alph_size
        for i in range(0, len(OUTPUT_ALPHABET)):
            for j in range(0, len(OUTPUT_ALPHABET)):
                cntr = 0
                for k in range(0, len(self.plaintext) - 2):
                    if self.plaintext[k] == OUTPUT_ALPHABET[i] and self.plaintext[k + 2] == OUTPUT_ALPHABET[j]:
                        cntr += 1
                bigram_ny_frequencies_plaintext[i * alph_size + j] = cntr / len(self.plaintext)

        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.plaintext_numberspace, 2, interval=2, recursive=False),
                              bigram_ny_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.ciphertext_numberspace, 2, interval=2, recursive=False),
                              bigram_ny_frequencies_ciphertext)

        # bigrams interval=7
        bigram_ny_frequencies_ciphertext = [0] * squared_alph_size
        for i in range(0, len(OUTPUT_ALPHABET)):
            for j in range(0, len(OUTPUT_ALPHABET)):
                cntr = 0
                for k in range(0, len(self.ciphertext) - 7):
                    if self.ciphertext[k] == OUTPUT_ALPHABET[i] and \
                            self.ciphertext[k + 7] == OUTPUT_ALPHABET[j]:
                        cntr += 1
                bigram_ny_frequencies_ciphertext[i * alph_size + j] = cntr / len(self.ciphertext)

        bigram_ny_frequencies_plaintext = [0] * squared_alph_size
        for i in range(0, len(OUTPUT_ALPHABET)):
            for j in range(0, len(OUTPUT_ALPHABET)):
                cntr = 0
                for k in range(0, len(self.plaintext) - 7):
                    if self.plaintext[k] == OUTPUT_ALPHABET[i] and self.plaintext[k + 7] == OUTPUT_ALPHABET[j]:
                        cntr += 1
                bigram_ny_frequencies_plaintext[i * alph_size + j] = cntr / len(self.plaintext)

        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.plaintext_numberspace, 2, interval=7, recursive=False),
                              bigram_ny_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.ciphertext_numberspace, 2, interval=7, recursive=False),
                              bigram_ny_frequencies_ciphertext)

        # trigrams interval=7
        trigram_ny_frequencies_ciphertext = [0] * third_pow_size
        for i in range(0, len(OUTPUT_ALPHABET)):
            for j in range(0, len(OUTPUT_ALPHABET)):
                for k in range(0, len(OUTPUT_ALPHABET)):
                    cntr = 0
                    for pos in range(0, len(self.ciphertext) - 14):
                        if self.ciphertext[pos] == OUTPUT_ALPHABET[i] and self.ciphertext[pos + 7] == OUTPUT_ALPHABET[j] and \
                                self.ciphertext[pos + 14] == OUTPUT_ALPHABET[k]:
                            cntr += 1
                    trigram_ny_frequencies_ciphertext[i * squared_alph_size + j * alph_size + k] = cntr / len(self.ciphertext)

        trigram_ny_frequencies_plaintext = [0] * third_pow_size
        for i in range(0, len(OUTPUT_ALPHABET)):
            for j in range(0, len(OUTPUT_ALPHABET)):
                for k in range(0, len(OUTPUT_ALPHABET)):
                    cntr = 0
                    for pos in range(0, len(self.plaintext) - 14):
                        if self.plaintext[pos] == OUTPUT_ALPHABET[i] and self.plaintext[pos + 7] == OUTPUT_ALPHABET[j]\
                                and self.plaintext[pos + 14] == OUTPUT_ALPHABET[k]:
                            cntr += 1
                    trigram_ny_frequencies_plaintext[i * squared_alph_size + j * alph_size + k] = cntr / len(self.plaintext)

        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.plaintext_numberspace, 3, interval=7, recursive=False),
                              trigram_ny_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.ciphertext_numberspace, 3, interval=7, recursive=False),
                              trigram_ny_frequencies_ciphertext)

        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.plaintext_numberspace, 3, interval=7, recursive=True),
                              bigram_ny_frequencies_plaintext + trigram_ny_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.ciphertext_numberspace, 3, interval=7, recursive=True),
                              bigram_ny_frequencies_ciphertext + trigram_ny_frequencies_ciphertext)

    def test03calculate_index_of_coincidence(self):
        self.assertEqual(round(ds.calculate_index_of_coincidence(self.ciphertext_numberspace), 3), 0.093)
        self.assertEqual(round(ds.calculate_index_of_coincidence(self.plaintext_numberspace), 3), 0.093)

    def test04calculate_digraphic_index_of_coincidence(self):
        self.assertEqual(round(ds.calculate_digraphic_index_of_coincidence(self.ciphertext_numberspace), 4), 0.0109)
        self.assertEqual(round(ds.calculate_digraphic_index_of_coincidence(self.plaintext_numberspace), 4), 0.0109)

    def test05has_letter_j(self):
        self.assertEqual(ds.has_letter_j(self.ciphertext_numberspace), self.ciphertext.decode().__contains__('j'))
        self.assertEqual(ds.has_letter_j(self.plaintext_numberspace), self.plaintext.decode().__contains__('j'))

    def test06calculate_chi_square(self):
        english_frequencies = [
            0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749,
            0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758, 0.00978, 0.0236, 0.0015, 0.01974, 0.00074]
        unigram_frequencies = [0]*26
        for c in self.cipher.alphabet:
            for i in range(0, len(self.ciphertext)):
                if self.ciphertext[i] == c:
                    unigram_frequencies[self.cipher.alphabet.index(c)] += 1
            unigram_frequencies[self.cipher.alphabet.index(c)] = unigram_frequencies[self.cipher.alphabet.index(c)] / len(self.ciphertext)
        chi_square = 0
        for i in range(0, len(unigram_frequencies)):
            residual = unigram_frequencies[i] - english_frequencies[i]
            chi_square += (residual * residual / english_frequencies[i])
        self.assertEqual(ds.calculate_chi_square(unigram_frequencies), chi_square / 100)

    def test07pattern_repetitions(self):
        self.assertEqual(ds.pattern_repetitions(np.array([0,0,1,1,2,2])), 2)
        self.assertEqual(ds.pattern_repetitions(np.array([0,0,0,1,1,2,2])), 7/3)
        self.assertEqual(ds.pattern_repetitions(np.array([0,0,1,0,1,2,2])), 2)
        self.assertEqual(ds.pattern_repetitions(np.array([0,1,2,3,0,1,2,3])), 0)
        self.assertEqual(ds.pattern_repetitions(np.array([0,0,0,0,1,1,2,2])), 8/3)
        self.assertEqual(ds.pattern_repetitions(np.array([0,0,0,0,0,1,1,2,2,0])), 3)
        self.assertEqual(ds.pattern_repetitions(np.array([0,0,0,0,0,0,1,1,2,2])), 3)
        self.assertEqual(ds.pattern_repetitions(np.array([0,0,0,0,0,0,0,0,0,0,1,1,2,2])), 14/4)

    def test08calculate_entropy(self):
        # https://stackoverflow.com/questions/2979174/how-do-i-compute-the-approximate-entropy-of-a-bit-string
        string = self.ciphertext.decode()
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]

        # calculate the entropy
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob]) / 10
        self.assertEqual(round(ds.calculate_entropy(self.ciphertext_numberspace), 6), round(entropy, 6))
        e = entropy

        string = self.plaintext.decode()
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]

        # calculate the entropy
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob]) / 10
        self.assertEqual(round(ds.calculate_entropy(self.plaintext_numberspace), 6), round(entropy, 6))
        self.assertEqual(e, entropy)

    def test9calculate_autocorrelation(self):
        # https://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python
        x = np.array(self.plaintext_numberspace)
        n = len(x)
        variance = x.var()
        x = x - x.mean()
        r = np.correlate(x, x, mode='full')[-n:]
        result = list(r / (variance * (np.arange(n, 0, -1))))
        result = result + [0]*(1000-len(result))
        self.assertEqual(ds.calculate_autocorrelation(np.array(self.plaintext_numberspace)), result)

        x = np.array(self.ciphertext_numberspace)
        n = len(x)
        variance = x.var()
        x = x - x.mean()
        r = np.correlate(x, x, mode='full')[-n:]
        result = list(r / (variance * (np.arange(n, 0, -1))))
        result = result + [0] * (1000 - len(result))
        self.assertEqual(ds.calculate_autocorrelation(np.array(self.ciphertext_numberspace)), result)

    def test10has_hash(self):
        no_route = b'fasdfasdfasdfasdfds'
        route = b'fsdfasddf#fasd#'
        no_route_ns = map_text_into_numberspace(no_route, OUTPUT_ALPHABET, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
        route_ns = map_text_into_numberspace(route, OUTPUT_ALPHABET, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
        self.assertEqual(ds.has_hash(no_route_ns), no_route.decode().__contains__('#'))
        self.assertEqual(ds.has_hash(route_ns), route.decode().__contains__('#'))

    def test11_calculate_maximum_index_of_coincidence(self):
        self.assertEqual(round(ds.calculate_maximum_index_of_coincidence(self.ciphertext_numberspace), 3), 0.116)
        self.assertEqual(round(ds.calculate_maximum_index_of_coincidence(self.plaintext_numberspace), 3), 0.116)
        self.assertEqual(ds.calculate_maximum_index_of_coincidence(self.ciphertext_numberspace), ds.calculate_maximum_index_of_coincidence(self.plaintext_numberspace))

    def test12_calculate_max_kappa(self):
        self.assertEqual(round(ds.calculate_max_kappa(self.ciphertext_numberspace), 3), 0.133)
        self.assertEqual(round(ds.calculate_max_kappa(self.plaintext_numberspace), 3), 0.133)
        self.assertEqual(ds.calculate_max_kappa(self.ciphertext_numberspace), ds.calculate_max_kappa(self.plaintext_numberspace))

    def test13_calculate_digraphic_index_of_coincidence_even(self):
        self.assertEqual(round(ds.calculate_digraphic_index_of_coincidence_even(self.ciphertext_numberspace), 4), 0.0106)
        self.assertEqual(round(ds.calculate_digraphic_index_of_coincidence_even(self.plaintext_numberspace), 4), 0.0106)
        self.assertEqual(round(ds.calculate_digraphic_index_of_coincidence_even(self.ciphertext_numberspace), 4),
            round(ds.calculate_digraphic_index_of_coincidence_even(self.plaintext_numberspace), 4))

    def test14_calculate_rod_lr(self):
        rod, lr = ds.calculate_rod_lr([0,0,0,0,1,1,1,1,2,2,2,2])
        self.assertEqual(round(lr, 3), 0.144)
        self.assertEqual(round(rod, 2), 0.67)

    def test15_is_dbl(self):
        self.assertEqual(ds.is_dbl([0,0,1,1]), 1)
        self.assertEqual(ds.is_dbl([1, 0, 1, 1]), 1)
        self.assertEqual(ds.is_dbl([0, 1, 1, 0]), 0)
        self.assertEqual(ds.is_dbl([0, 0, 1]), 0)

    def test16_calculate_max_nicodemus_ic(self):
        self.assertEqual(round(ds.calculate_nic(self.plaintext_numberspace), 3), 0.103)
        self.assertEqual(round(ds.calculate_nic(self.ciphertext_numberspace), 3), 0.103)
        self.assertEqual(round(ds.calculate_nic(self.plaintext_numberspace), 3), round(ds.calculate_nic(self.ciphertext_numberspace), 3))

    def test17_calculate_sdd(self):
        self.assertEqual(round(ds.calculate_sdd(self.plaintext_numberspace), 3), 0.292)
        self.assertEqual(round(ds.calculate_sdd(self.ciphertext_numberspace), 3), 0.094)

    def test18_calculate_ldi_stats(self):
        ldi_stats = ds.calculate_ldi_stats(self.ciphertext_numberspace)
        ldi_stats = [round(x, 3) for x in ldi_stats]
        self.assertEqual(ldi_stats, [0.755, 0.760, 0.701, 0.760, 0.746])
        ldi_stats = ds.calculate_ldi_stats(self.plaintext_numberspace)
        ldi_stats = [round(x, 3) for x in ldi_stats]
        self.assertEqual(ldi_stats, [0.751, 0.763, 0.728, 0.782, 0.799])

    def test19_calculate_phic(self):
        self.assertEqual(round(ds.calculate_phic(self.ciphertext_numberspace), 3), 0.861)
        self.assertEqual(round(ds.calculate_phic(self.plaintext_numberspace), 3), 0.861)
        self.assertEqual(round(ds.calculate_phic(self.ciphertext_numberspace), 3), round(ds.calculate_phic(self.plaintext_numberspace), 3))


    '''The methods calculate_statistics and encrypt can not be tested properly, because they are either random or are only depending on
    other methods'''