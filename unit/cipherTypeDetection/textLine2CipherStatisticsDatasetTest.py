import unittest
from collections import Counter
from cipherTypeDetection import textLine2CipherStatisticsDataset as ds
import unit.cipherImplementations.cipherTest as cipherTest
import util.textUtils as text_utils
import math
import numpy as np


class TextLine2CipherStatisticsDatasetTest(unittest.TestCase):
    cipher = cipherTest.CipherTest.cipher
    ALPHABET = text_utils.map_text_into_numberspace(cipher.alphabet, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'thisisafilteredplaintextwithsomewordsinittobeusedtotestthestatisticsofthetextlinetocipherstatisticsd'
    plaintext_numberspace = text_utils.map_text_into_numberspace(plaintext, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
    # key = fcghartokldibuezjpqxyvwnsm
    # cipher: simple substitution
    ciphertext = b'xokqkqfrkixapahzifkuxanxwkxoqebawephqkukxxecayqahxexaqxxoaqxfxkqxkgqerxoaxanxikuaxegkzoapqxfxkqxkgqh'
    ciphertext_numberspace = text_utils.map_text_into_numberspace(ciphertext, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)

    def test01calculate_frequencies(self):
        # unigrams
        plaintext_counter = Counter(self.plaintext.decode())
        ciphertext_counter = Counter(self.ciphertext.decode())
        unigram_frequencies_plaintext = [0]*26
        unigram_frequencies_ciphertext = [0]*26
        for i, c in enumerate(self.cipher.alphabet.decode()):
            unigram_frequencies_plaintext[i] = plaintext_counter[c] / len(self.plaintext)
            unigram_frequencies_ciphertext[i] = ciphertext_counter[c] / len(self.ciphertext)

        self.assertCountEqual(ds.calculate_frequencies(self.plaintext_numberspace, 1, recursive=False), unigram_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_frequencies(self.ciphertext_numberspace, 1, recursive=False), unigram_frequencies_ciphertext)

        keys_plaintext = list(plaintext_counter.keys())
        keys_simple_substitution_cipher = list(ciphertext_counter.keys())
        for i in range(0, len(keys_plaintext)):
            self.assertEqual(plaintext_counter[keys_plaintext[i]], ciphertext_counter[keys_simple_substitution_cipher[i]])

        # bigrams
        bigram_frequencies_ciphertext = [0]*676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                cntr = 0
                for k in range(0, len(self.ciphertext) - 1):
                    if self.ciphertext[k] == self.cipher.alphabet[i] and \
                            self.ciphertext[k + 1] == self.cipher.alphabet[j]:
                        cntr += 1
                bigram_frequencies_ciphertext[i*26+j] = cntr / len(self.ciphertext)

        bigram_frequencies_plaintext = [0]*676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                cntr = 0
                for k in range(0, len(self.plaintext)-1):
                    if self.plaintext[k] == self.cipher.alphabet[i] and \
                            self.plaintext[k+1] == self.cipher.alphabet[j]:
                        cntr += 1
                bigram_frequencies_plaintext[i*26+j] = cntr / len(self.plaintext)

        self.assertCountEqual(ds.calculate_frequencies(self.plaintext_numberspace, 2, recursive=False), bigram_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_frequencies(self.ciphertext_numberspace, 2, recursive=False), bigram_frequencies_ciphertext)

        # trigrams
        trigram_frequencies_ciphertext = [0]*17576
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                for k in range(0, len(self.cipher.alphabet)):
                    cntr = 0
                    for pos in range(0, len(self.ciphertext) - 2):
                        if self.ciphertext[pos] == self.cipher.alphabet[i] and self.ciphertext[pos + 1] == self.cipher.alphabet[j]\
                                and self.ciphertext[pos + 2] == self.cipher.alphabet[k]:
                            cntr += 1
                    trigram_frequencies_ciphertext[i * 676 + j * 26 + k] = cntr / len(self.ciphertext)

        trigram_frequencies_plaintext = [0]*17576
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                for k in range(0, len(self.cipher.alphabet)):
                    cntr = 0
                    for pos in range(0, len(self.plaintext) - 2):
                        if self.plaintext[pos] == self.cipher.alphabet[i] and self.plaintext[pos + 1] == self.cipher.alphabet[j]\
                                and self.plaintext[pos + 2] == self.cipher.alphabet[k]:
                            cntr += 1
                    trigram_frequencies_plaintext[i * 676 + j * 26 + k] = cntr / len(self.ciphertext)

        self.assertCountEqual(ds.calculate_frequencies(self.plaintext_numberspace, 3, recursive=False), trigram_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_frequencies(self.ciphertext_numberspace, 3, recursive=False), trigram_frequencies_ciphertext)

        self.assertCountEqual(ds.calculate_frequencies(self.plaintext_numberspace, 3, recursive=True),
                              unigram_frequencies_plaintext + bigram_frequencies_plaintext + trigram_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_frequencies(self.ciphertext_numberspace, 3, recursive=True),
                              unigram_frequencies_ciphertext + bigram_frequencies_ciphertext + trigram_frequencies_ciphertext)

    def test02calculate_ny_gram_frequencies(self):
        # bigrams interval=2
        bigram_ny_frequencies_ciphertext = [0] * 676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                cntr = 0
                for k in range(0, len(self.ciphertext) - 2):
                    if self.ciphertext[k] == self.cipher.alphabet[i] and \
                            self.ciphertext[k + 2] == self.cipher.alphabet[j]:
                        cntr += 1
                bigram_ny_frequencies_ciphertext[i * 26 + j] = cntr / len(self.ciphertext)

        bigram_ny_frequencies_plaintext = [0] * 676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                cntr = 0
                for k in range(0, len(self.plaintext) - 2):
                    if self.plaintext[k] == self.cipher.alphabet[i] and self.plaintext[k + 2] == self.cipher.alphabet[j]:
                        cntr += 1
                bigram_ny_frequencies_plaintext[i * 26 + j] = cntr / len(self.plaintext)

        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.plaintext_numberspace, 2, interval=2, recursive=False),
                              bigram_ny_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.ciphertext_numberspace, 2, interval=2, recursive=False),
                              bigram_ny_frequencies_ciphertext)

        # bigrams interval=7
        bigram_ny_frequencies_ciphertext = [0] * 676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                cntr = 0
                for k in range(0, len(self.ciphertext) - 7):
                    if self.ciphertext[k] == self.cipher.alphabet[i] and \
                            self.ciphertext[k + 7] == self.cipher.alphabet[j]:
                        cntr += 1
                bigram_ny_frequencies_ciphertext[i * 26 + j] = cntr / len(self.ciphertext)

        bigram_ny_frequencies_plaintext = [0] * 676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                cntr = 0
                for k in range(0, len(self.plaintext) - 7):
                    if self.plaintext[k] == self.cipher.alphabet[i] and self.plaintext[k + 7] == self.cipher.alphabet[j]:
                        cntr += 1
                bigram_ny_frequencies_plaintext[i * 26 + j] = cntr / len(self.plaintext)

        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.plaintext_numberspace, 2, interval=7, recursive=False),
                              bigram_ny_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.ciphertext_numberspace, 2, interval=7, recursive=False),
                              bigram_ny_frequencies_ciphertext)

        # trigrams interval=7
        trigram_ny_frequencies_ciphertext = [0] * 17576
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                for k in range(0, len(self.cipher.alphabet)):
                    cntr = 0
                    for pos in range(0, len(self.ciphertext) - 14):
                        if self.ciphertext[pos] == self.cipher.alphabet[i] and self.ciphertext[pos + 7] == self.cipher.alphabet[j] and \
                                self.ciphertext[pos + 14] == self.cipher.alphabet[k]:
                            cntr += 1
                    trigram_ny_frequencies_ciphertext[i * 676 + j * 26 + k] = cntr / len(self.ciphertext)

        trigram_ny_frequencies_plaintext = [0] * 17576
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                for k in range(0, len(self.cipher.alphabet)):
                    cntr = 0
                    for pos in range(0, len(self.plaintext) - 14):
                        if self.plaintext[pos] == self.cipher.alphabet[i] and self.plaintext[pos + 7] == self.cipher.alphabet[j]\
                                and self.plaintext[pos + 14] == self.cipher.alphabet[k]:
                            cntr += 1
                    trigram_ny_frequencies_plaintext[i * 676 + j * 26 + k] = cntr / len(self.plaintext)

        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.plaintext_numberspace, 3, interval=7, recursive=False),
                              trigram_ny_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.ciphertext_numberspace, 3, interval=7, recursive=False),
                              trigram_ny_frequencies_ciphertext)

        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.plaintext_numberspace, 3, interval=7, recursive=True),
                              bigram_ny_frequencies_plaintext + trigram_ny_frequencies_plaintext)
        self.assertCountEqual(ds.calculate_ny_gram_frequencies(self.ciphertext_numberspace, 3, interval=7, recursive=True),
                              bigram_ny_frequencies_ciphertext + trigram_ny_frequencies_ciphertext)

    def test03calculate_index_of_coincidence(self):
        n = [0]*26
        for c in self.cipher.alphabet:
            for i in range(0, len(self.ciphertext)):
                if self.ciphertext[i] == c:
                    n[self.cipher.alphabet.index(c)] += 1
        ic = 0
        for i in range(0, len(n)):
            ic += n[i] * (n[i] - 1)
        ic = ic / len(self.ciphertext)
        ic = ic / (len(self.ciphertext) - 1)
        self.assertEqual(ds.calculate_index_of_coincidence(self.ciphertext_numberspace), ic)

        n = [0] * 26
        for c in self.cipher.alphabet:
            for i in range(0, len(self.plaintext)):
                if self.plaintext[i] == c:
                    n[self.cipher.alphabet.index(c)] += 1
        ic = 0
        for i in range(0, len(n)):
            ic += n[i] * (n[i] - 1)
        ic = ic / len(self.plaintext)
        ic = ic / (len(self.plaintext) - 1)
        self.assertEqual(ds.calculate_index_of_coincidence(
            self.plaintext_numberspace), ic)

    def test04calculate_index_of_coincidence_bigrams(self):
        n = [0]*676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                for k in range(0, len(self.ciphertext) - 1):
                    if self.ciphertext[k] == self.cipher.alphabet[i] and self.ciphertext[k + 1] == self.cipher.alphabet[j]:
                        n[i * 26 + j] += 1
        ic = 0
        for i in range(0, len(n)):
            ic += n[i] * (n[i] - 1)
        ic = ic / len(self.ciphertext)
        ic = ic / (len(self.ciphertext) - 1)
        ic = ic / (len(self.ciphertext) - 2)
        self.assertEqual(ds.calculate_index_of_coincidence_bigrams(self.ciphertext_numberspace), ic)

        n = [0] * 676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                for k in range(0, len(self.plaintext) - 1):
                    if self.plaintext[k] == self.cipher.alphabet[i] and \
                            self.plaintext[k + 1] == self.cipher.alphabet[j]:
                        n[i * 26 + j] += 1
        ic = 0
        for i in range(0, len(n)):
            ic += n[i] * (n[i] - 1)
        ic = ic / len(self.plaintext)
        ic = ic / (len(self.plaintext) - 1)
        ic = ic / (len(self.plaintext) - 2)
        self.assertEqual(ds.calculate_index_of_coincidence_bigrams(self.plaintext_numberspace), ic)

    def test05has_letter_j(self):
        self.assertEqual(ds.has_letter_j(self.ciphertext_numberspace), self.ciphertext.decode().__contains__('j'))
        self.assertEqual(ds.has_letter_j(self.plaintext_numberspace), self.plaintext.decode().__contains__('j'))

    def test06has_doubles(self):
        has_doubles = 0
        for i in range(0, len(self.ciphertext) - 1):
            if self.ciphertext[i] == self.ciphertext[i + 1]:
                has_doubles = 1
        self.assertEqual(ds.has_doubles(self.ciphertext_numberspace), has_doubles)

        has_doubles = 0
        for i in range(0, len(self.plaintext) - 1):
            if self.plaintext[i] == self.plaintext[i + 1]:
                has_doubles = 1
        self.assertEqual(ds.has_doubles(self.plaintext_numberspace), has_doubles)

    def test07calculate_chi_square(self):
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
        self.assertEqual(ds.calculate_chi_square(unigram_frequencies), chi_square)

    def test08pattern_repetitions(self):
        # count patterns of 3
        counter = 0
        text = self.ciphertext.decode()
        patterns = []
        for i in range(0, len(self.ciphertext) - 2):
            pattern = text[i] + text[i+1] + text[i+2]
            if pattern not in patterns:
                patterns.append(pattern)
                for j in range(i+1, len(self.ciphertext) - 2):
                    if pattern == text[j] + text[j+1] + text[j+2]:
                        counter += 1
        self.assertEqual(ds.pattern_repetitions(self.ciphertext_numberspace), counter)

    def test09calculate_entropy(self):
        # https://stackoverflow.com/questions/2979174/how-do-i-compute-the-approximate-entropy-of-a-bit-string
        string = self.ciphertext.decode()
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]

        # calculate the entropy
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
        self.assertEqual(round(ds.calculate_entropy(self.ciphertext_numberspace), 6), round(entropy, 6))
        e = entropy

        string = self.plaintext.decode()
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]

        # calculate the entropy
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
        self.assertEqual(round(ds.calculate_entropy(self.plaintext_numberspace), 6), round(entropy, 6))
        self.assertEqual(e, entropy)

    def test10calculate_autocorrelation_average(self):
        # https://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python
        x = self.plaintext_numberspace
        n = len(x)
        variance = x.var()
        x = x - x.mean()
        r = np.correlate(x, x, mode='full')[-n:]
        result = r / (variance * (np.arange(n, 0, -1)))
        avg = 0
        for i in range(0, len(result)):
            avg += result[i]
        avg = avg / len(result)
        self.assertEqual(ds.calculate_autocorrelation_average(self.plaintext_numberspace), avg)

        x = self.ciphertext_numberspace
        n = len(x)
        variance = x.var()
        x = x - x.mean()
        r = np.correlate(x, x, mode='full')[-n:]
        result = r / (variance * (np.arange(n, 0, -1)))
        avg = 0
        for i in range(0, len(result)):
            avg += result[i]
        avg = avg / len(result)
        self.assertEqual(ds.calculate_autocorrelation_average(self.ciphertext_numberspace), avg)

    '''The methods calculate_statistics and encrypt can not be tested properly, because they are either random or are only depending on
    other methods'''