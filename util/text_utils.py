import numpy as np

#maps a given string into number space, i.e. a number array
#using the defined alphabet
def map_text_into_numberspace(text, alphabet, unknown_symbol_number) :
    array = []
    for c in text:
        if bytes([c]) in alphabet:
            array.append(alphabet.index(bytes([c])))
        else :
            array.append(unknown_symbol_number)
    return np.array(array)

#maps a given number array into text space, i.e. a string
#using the defined alphabet
def map_numbers_into_textspace(numbers, alphabet, unknown_symbol) :
    output = b''
    for n in numbers :
        if n > len(alphabet) :
            output = output + unknown_symbol
        else :
            output = output + bytes([alphabet[n]])
    return output

def remove_unknown_symbols(text, alphabet):
    i = 0
    while i < len(text):
        if text[i] not in alphabet:
            text = text.replace(bytes([text[i]]), b'')
        else:
            i += 1
    return text

def num_index_of(array, value):
    for position in range(0, len(array)):
        if array[position] == value:
            return position
    return -1
