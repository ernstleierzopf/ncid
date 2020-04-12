import unittest
from collections import Counter
from cipherTypeDetection import textLine2CipherStatisticsDataset as textLine2CipherStatisticsDataset
import unit.cipherImplementations.cipherTest as cipherTest
import util.textUtils as text_utils
import math


class TextLine2CipherStatisticsDatasetTest(unittest.TestCase):
    cipher = cipherTest.CipherTest.cipher
    ALPHABET = text_utils.map_text_into_numberspace(cipher.alphabet, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'thisisafilteredplaintextwithsomewordsinittobeusedtotestthestatisticsofthetextlinetocipherstatisticsd'
    plaintext_numberspace = text_utils.map_text_into_numberspace(plaintext, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
    # key = fcghartokldibuezjpqxyvwnsm
    simple_substitution_ciphertext = b'xokqkqfrkixapahzifkuxanxwkxoqebawephqkukxxecayqahxexaqxxoaqxfxkqxkgqerxoaxanxikuaxegkzoapqxfxkqxkgqh'
    simple_substitution_ciphertext_numberspace = text_utils.map_text_into_numberspace(simple_substitution_ciphertext, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)

    def test01calculate_frequencies(self):
        # unigrams
        plaintext_counter = Counter(self.plaintext.decode())
        simple_substitution_ciphertext_counter = Counter(self.simple_substitution_ciphertext.decode())
        unigram_frequencies_plaintext = [0]*26
        unigram_frequencies_simple_substitution_ciphertext = [0]*26
        for i, c in enumerate(self.cipher.alphabet.decode()):
            unigram_frequencies_plaintext[i] = plaintext_counter[c] / len(self.plaintext)
            unigram_frequencies_simple_substitution_ciphertext[i] = simple_substitution_ciphertext_counter[c] / len(self.simple_substitution_ciphertext)

        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_frequencies(self.plaintext_numberspace, 1,
            recursive=False), unigram_frequencies_plaintext)
        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_frequencies(self.simple_substitution_ciphertext_numberspace, 1,
            recursive=False), unigram_frequencies_simple_substitution_ciphertext)

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

        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_frequencies(self.plaintext_numberspace, 2,
            recursive=False), bigram_frequencies_plaintext)
        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_frequencies(self.simple_substitution_ciphertext_numberspace, 2,
            recursive=False), bigram_frequencies_simple_substitution_ciphertext)

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

        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_frequencies(self.plaintext_numberspace, 3,
            recursive=False), trigram_frequencies_plaintext)
        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_frequencies(self.simple_substitution_ciphertext_numberspace, 3,
            recursive=False), trigram_frequencies_simple_substitution_ciphertext)

        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_frequencies(self.plaintext_numberspace, 3,
            recursive=True), unigram_frequencies_plaintext + bigram_frequencies_plaintext + trigram_frequencies_plaintext)
        self.assertCountEqual(
            textLine2CipherStatisticsDataset.calculate_frequencies(self.simple_substitution_ciphertext_numberspace, 3,
                recursive=True), unigram_frequencies_simple_substitution_ciphertext + bigram_frequencies_simple_substitution_ciphertext + trigram_frequencies_simple_substitution_ciphertext)

    def test02calculate_ny_gram_frequencies(self):
        # bigrams interval=2
        bigram_ny_frequencies_simple_substitution_ciphertext = [0] * 676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                cntr = 0
                for k in range(0, len(self.simple_substitution_ciphertext) - 2):
                    if self.simple_substitution_ciphertext[k] == self.cipher.alphabet[i] and \
                            self.simple_substitution_ciphertext[k + 2] == self.cipher.alphabet[j]:
                        cntr += 1
                bigram_ny_frequencies_simple_substitution_ciphertext[i * 26 + j] = cntr / len(self.simple_substitution_ciphertext)

        bigram_ny_frequencies_plaintext = [0] * 676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                cntr = 0
                for k in range(0, len(self.plaintext) - 2):
                    if self.plaintext[k] == self.cipher.alphabet[i] and self.plaintext[k + 2] == self.cipher.alphabet[j]:
                        cntr += 1
                bigram_ny_frequencies_plaintext[i * 26 + j] = cntr / len(self.plaintext)

        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_ny_gram_frequencies(self.plaintext_numberspace, 2,
            interval=2, recursive=False), bigram_ny_frequencies_plaintext)
        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_ny_gram_frequencies(self.simple_substitution_ciphertext_numberspace, 2,
            interval=2, recursive=False), bigram_ny_frequencies_simple_substitution_ciphertext)

        # bigrams interval=7
        bigram_ny_frequencies_simple_substitution_ciphertext = [0] * 676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                cntr = 0
                for k in range(0, len(self.simple_substitution_ciphertext) - 7):
                    if self.simple_substitution_ciphertext[k] == self.cipher.alphabet[i] and \
                            self.simple_substitution_ciphertext[k + 7] == self.cipher.alphabet[j]:
                        cntr += 1
                bigram_ny_frequencies_simple_substitution_ciphertext[i * 26 + j] = cntr / len(self.simple_substitution_ciphertext)

        bigram_ny_frequencies_plaintext = [0] * 676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                cntr = 0
                for k in range(0, len(self.plaintext) - 7):
                    if self.plaintext[k] == self.cipher.alphabet[i] and self.plaintext[k + 7] == self.cipher.alphabet[j]:
                        cntr += 1
                bigram_ny_frequencies_plaintext[i * 26 + j] = cntr / len(self.plaintext)

        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_ny_gram_frequencies(self.plaintext_numberspace, 2,
            interval=7, recursive=False), bigram_ny_frequencies_plaintext)
        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_ny_gram_frequencies(
            self.simple_substitution_ciphertext_numberspace, 2, interval=7, recursive=False), bigram_ny_frequencies_simple_substitution_ciphertext)

        # trigrams interval=7
        trigram_ny_frequencies_simple_substitution_ciphertext = [0] * 17576
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                for k in range(0, len(self.cipher.alphabet)):
                    cntr = 0
                    for l in range(0, len(self.simple_substitution_ciphertext) - 14):
                        if self.simple_substitution_ciphertext[l] == self.cipher.alphabet[i] and \
                                self.simple_substitution_ciphertext[l + 7] == self.cipher.alphabet[j] and \
                                self.simple_substitution_ciphertext[l + 14] == self.cipher.alphabet[k]:
                            cntr += 1
                    trigram_ny_frequencies_simple_substitution_ciphertext[i * 676 + j * 26 + k] = cntr / len(
                        self.simple_substitution_ciphertext)

        trigram_ny_frequencies_plaintext = [0] * 17576
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                for k in range(0, len(self.cipher.alphabet)):
                    cntr = 0
                    for l in range(0, len(self.plaintext) - 14):
                        if self.plaintext[l] == self.cipher.alphabet[i] and self.plaintext[l + 7] == self.cipher.alphabet[j]\
                            and self.plaintext[l + 14] == self.cipher.alphabet[k]:
                            cntr += 1
                    trigram_ny_frequencies_plaintext[i * 676 + j * 26 + k] = cntr / len(self.plaintext)

        self.assertCountEqual(
            textLine2CipherStatisticsDataset.calculate_ny_gram_frequencies(self.plaintext_numberspace, 3, interval=7,
                recursive=False), trigram_ny_frequencies_plaintext)
        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_ny_gram_frequencies(
            self.simple_substitution_ciphertext_numberspace, 3, interval=7, recursive=False),
            trigram_ny_frequencies_simple_substitution_ciphertext)

        self.assertCountEqual(
            textLine2CipherStatisticsDataset.calculate_ny_gram_frequencies(self.plaintext_numberspace, 3, interval=7,
                recursive=True), bigram_ny_frequencies_plaintext + trigram_ny_frequencies_plaintext)
        self.assertCountEqual(textLine2CipherStatisticsDataset.calculate_ny_gram_frequencies(
            self.simple_substitution_ciphertext_numberspace, 3, interval=7, recursive=True),
            bigram_ny_frequencies_simple_substitution_ciphertext + trigram_ny_frequencies_simple_substitution_ciphertext)


    def test03calculate_index_of_coincidence(self):
        n = [0]*26
        for c in self.cipher.alphabet:
            for i in range(0, len(self.simple_substitution_ciphertext)):
                if self.simple_substitution_ciphertext[i] == c:
                    n[self.cipher.alphabet.index(c)] += 1
        ic = 0
        for i in range(0, len(n)):
            ic += n[i] * (n[i] - 1)
        ic = ic / len(self.simple_substitution_ciphertext)
        ic = ic / (len(self.simple_substitution_ciphertext) - 1)
        self.assertEqual(textLine2CipherStatisticsDataset.calculate_index_of_coincidence(self.simple_substitution_ciphertext_numberspace), ic)

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
        self.assertEqual(textLine2CipherStatisticsDataset.calculate_index_of_coincidence(
            self.plaintext_numberspace), ic)

    def test04calculate_index_of_coincidence_bigrams(self):
        n = [0]*676
        for i in range(0, len(self.cipher.alphabet)):
            for j in range(0, len(self.cipher.alphabet)):
                for k in range(0, len(self.simple_substitution_ciphertext) - 1):
                    if self.simple_substitution_ciphertext[k] == self.cipher.alphabet[i] and \
                        self.simple_substitution_ciphertext[k + 1] == self.cipher.alphabet[j]:
                            n[i * 26 + j] += 1
        ic = 0
        for i in range(0, len(n)):
            ic += n[i] * (n[i] - 1)
        ic = ic / len(self.simple_substitution_ciphertext)
        ic = ic / (len(self.simple_substitution_ciphertext) - 1)
        ic = ic / (len(self.simple_substitution_ciphertext) - 2)
        self.assertEqual(textLine2CipherStatisticsDataset.calculate_index_of_coincidence_bigrams(self.simple_substitution_ciphertext_numberspace), ic)

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
        self.assertEqual(textLine2CipherStatisticsDataset.calculate_index_of_coincidence_bigrams(
            self.plaintext_numberspace), ic)


    def test05has_letter_j(self):
        self.assertEqual(textLine2CipherStatisticsDataset.has_letter_j(self.simple_substitution_ciphertext_numberspace),
            self.simple_substitution_ciphertext.decode().__contains__('j'))
        self.assertEqual(textLine2CipherStatisticsDataset.has_letter_j(self.plaintext_numberspace),
            self.plaintext.decode().__contains__('j'))

    def test06has_doubles(self):
        has_doubles = 0
        for i in range(0, len(self.simple_substitution_ciphertext) - 1):
            if self.simple_substitution_ciphertext[i] == self.simple_substitution_ciphertext[i+1]:
                has_doubles = 1
        self.assertEqual(textLine2CipherStatisticsDataset.has_doubles(self.simple_substitution_ciphertext_numberspace), has_doubles)

        has_doubles = 0
        for i in range(0, len(self.plaintext) - 1):
            if self.plaintext[i] == self.plaintext[i + 1]:
                has_doubles = 1
        self.assertEqual(textLine2CipherStatisticsDataset.has_doubles(self.plaintext_numberspace), has_doubles)

    def test07calculate_chi_square(self):
        english_frequencies = [0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094, 0.06966, 0.00153,
            0.00772, 0.04025, 0.02406, 0.06749, 0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758, 0.00978,
            0.0236, 0.0015, 0.01974, 0.00074]
        unigram_frequencies = [0]*26
        for c in self.cipher.alphabet:
            for i in range(0, len(self.simple_substitution_ciphertext)):
                if self.simple_substitution_ciphertext[i] == c:
                    unigram_frequencies[self.cipher.alphabet.index(c)] += 1
            unigram_frequencies[self.cipher.alphabet.index(c)] = unigram_frequencies[self.cipher.alphabet.index(c)] / len(self.simple_substitution_ciphertext)
        chi_square = 0
        for i in range(0, len(unigram_frequencies)):
            residual = unigram_frequencies[i] - english_frequencies[i]
            chi_square += (residual * residual / english_frequencies[i])
        self.assertEqual(textLine2CipherStatisticsDataset.calculate_chi_square(unigram_frequencies), chi_square)

    def test08pattern_repetitions(self):
        # count patterns of 3
        counter = 0
        text = self.simple_substitution_ciphertext.decode()
        patterns = []
        for i in range(0, len(self.simple_substitution_ciphertext) - 2):
            pattern = text[i] + text[i+1] + text[i+2]
            if pattern not in patterns:
                patterns.append(pattern)
                for j in range(i+1, len(self.simple_substitution_ciphertext) - 2):
                    if pattern == text[j] + text[j+1] + text[j+2]:
                        counter += 1
        self.assertEqual(textLine2CipherStatisticsDataset.pattern_repetitions(self.simple_substitution_ciphertext_numberspace), counter)

    def test09calculate_entropy(self):
        string = self.simple_substitution_ciphertext.decode()
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]

        # calculate the entropy
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
        self.assertEqual(round(textLine2CipherStatisticsDataset.calculate_entropy(self.simple_substitution_ciphertext_numberspace), 6), round(entropy, 6))
        e = entropy

        string = self.plaintext.decode()
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]

        # calculate the entropy
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
        self.assertEqual(round(textLine2CipherStatisticsDataset.calculate_entropy(self.plaintext_numberspace), 6), round(entropy, 6))
        self.assertEqual(e, entropy)

    # def test10calculate_autocorrelation(self):
    #     pass

    '''
    The methods calculate_statistics and encrypt can not be tested properly, because they are either random or are only depending on other methods
    '''