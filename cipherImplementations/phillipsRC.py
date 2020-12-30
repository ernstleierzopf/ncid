from cipherImplementations.phillips import Phillips


class PhillipsRC(Phillips):
    def shift_key(self, i, key, key_shift_cntr, new_key):
        row_shift = 5 * key_shift_cntr
        tmp = new_key[row_shift:row_shift + 5]
        new_key[row_shift:row_shift + 5] = new_key[row_shift + 5:row_shift + 10]
        new_key[row_shift + 5:row_shift + 10] = tmp

        tmp = []
        for j in range(5):
            tmp.append(new_key[key_shift_cntr + j * 5])

        for j in range(5):
            new_key[key_shift_cntr + j * 5] = new_key[key_shift_cntr + j * 5 + 1]
            new_key[key_shift_cntr + j * 5 + 1] = tmp[j]

        key_shift_cntr += 1
        if key_shift_cntr == 4:
            key_shift_cntr = 0
        if i % 40 == 0:
            new_key = list(key)
        return new_key, key_shift_cntr
