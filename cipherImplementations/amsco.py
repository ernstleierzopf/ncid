from cipherImplementations.cipher import Cipher, generate_random_list_of_unique_digits
import numpy as np


def calc_column_number(key_length, string_length, char_num):
    col_num = [0] * key_length
    index_col = 0
    counter = string_length
    while counter > 0:
        col_num[index_col] += char_num
        counter -= char_num
        index_col = (index_col + 1) % key_length
        if char_num == 1:
            char_num = 2
        else:
            char_num = 1
        if counter == 1 and char_num == 2:  # if true we are at the last letter
            char_num = 1  # so only add one to the count
    return col_num


class Amsco(Cipher):
    """Adapted implementation from https://github.com/csmith50/AMSCO_Cipher"""

    def __init__(self, alphabet, unknown_symbol, unknown_symbol_number, start_with_letter_count=2):
        if not 0 < start_with_letter_count < 3:
            raise ValueError('start_with_letter_count must be 1 or 2!')
        self.char_num = start_with_letter_count
        self.alphabet = alphabet
        self.unknown_symbol = unknown_symbol
        self.unknown_symbol_number = unknown_symbol_number

    def generate_random_key(self, length):
        if not isinstance(length, int) or not 1 < length < 10:
            raise ValueError('The AMSCO cipher can handle only keys with the length 2-9.')
        return generate_random_list_of_unique_digits(length)

    def encrypt(self, plaintext, key):
        # setup: get the number of characters in each column
        key_length = len(key)  # length of key determines number of columns
        string_length = len(plaintext)
        col_num = calc_column_number(key_length, string_length, self.char_num)

        # order columns and match with the number of characters
        digits_with_chars = [(digit, col_num[i]) for i, digit in enumerate(key)]

        # chop up text
        chopped = [[] for _ in range(key_length)]
        counter = string_length
        key_counter = 0
        while counter > 0:  # works just like calc_col_number except with chars
            number, _col_number = digits_with_chars[key_counter]
            for i in range(self.char_num):
                chopped[int(number) - 1].append(plaintext[i])
            plaintext = plaintext[self.char_num:]
            counter -= self.char_num
            key_counter = (key_counter + 1) % key_length
            if self.char_num == 1:
                self.char_num = 2
            else:
                self.char_num = 1
            if self.char_num == 2 and counter == 1:
                self.char_num = 1

        # put ciphter text together
        ciphertext = []
        for i in range(key_length):
            ciphertext += chopped[i]
        return np.array(ciphertext)

    def decrypt(self, ciphertext, key):
        # setup
        key_length = len(key)
        string_length = len(ciphertext)
        col_num = calc_column_number(key_length, string_length, self.char_num)

        # order columns and match with the number of letters in each
        digits_with_chars = [(digit, col_num[i]) for i, digit in enumerate(key)]
        digits_with_chars = sorted(digits_with_chars)

        # chop up text
        chopped = [[] for _ in range(key_length)]
        for i in range(key_length):
            _digit, col_number = digits_with_chars[i]
            for j in range(col_number):
                chopped[i].append(ciphertext[j])
            ciphertext = ciphertext[col_number:]

        # put decrypted message together
        key_counter = 0
        counter = string_length
        plaintext = []
        while counter > 0:
            index_column = int(key[key_counter]) - 1
            for i in range(self.char_num):
                if len(chopped[index_column]) > i:
                    plaintext.append(chopped[index_column][i])
            chopped[index_column] = chopped[index_column][self.char_num:]
            counter -= self.char_num
            key_counter = (key_counter + 1) % key_length
            if self.char_num == 1:
                self.char_num = 2
            else:
                self.char_num = 1
            if self.char_num == 2 and counter == 1:
                self.char_num = 1
        return np.array(plaintext)
