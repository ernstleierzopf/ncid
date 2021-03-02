import sys
import os
from textUtils import map_text_into_numberspace
sys.path.append("../")
from cipherTypeDetection.config import CIPHER_TYPES, CIPHER_IMPLEMENTATIONS
from cipherTypeDetection.textLine2CipherStatisticsDataset import calculate_statistics
from cipherImplementations.cipher import OUTPUT_ALPHABET
types = CIPHER_TYPES

specialized_model = True
if specialized_model:
    types = ['amsco', 'bazeries', 'beaufort', 'bifid', 'cmbifid', 'digrafid', 'foursquare', 'fractionated_morse', 'gromark', 'gronsfeld',
             'homophonic', 'monome_dinome', 'morbit', 'myszkowski', 'nicodemus', 'nihilist_substitution', 'periodic_gromark', 'phillips',
             'playfair', 'pollux', 'porta', 'portax', 'progressive_key', 'quagmire2', 'quagmire3', 'quagmire4', 'ragbaby', 'redefence',
             'seriated_playfair', 'slidefair', 'swagman', 'tridigital', 'trifid', 'tri_square', 'two_square', 'vigenere']

with open('../data/aca_ciphertexts.txt', 'rb') as fd:
    lines = fd.readlines()
fd = os.open('../data/aca_features.txt', os.O_WRONLY | os.O_CREAT)
for line in lines:
    line = line.strip(b'\n')
    cipher_type, ciphertext = line.split(b' ', 1)
    cipher_type = cipher_type.decode()
    label = types.index(cipher_type)
    cipher = CIPHER_IMPLEMENTATIONS[label]
    ciphertext_numberspace = map_text_into_numberspace(ciphertext, OUTPUT_ALPHABET, cipher.unknown_symbol_number)
    features = calculate_statistics(ciphertext_numberspace)
    print(cipher_type, label, ciphertext, features)
    arr = bytearray()
    for f in features:
        arr += b'%f,' % f
    arr = bytes(arr)[:-1]
    cipher = bytearray()
    for c in ciphertext_numberspace:
        cipher += b'%d,' % c
    cipher = bytes(cipher)[:-1]
    os.write(fd, b'%d %s %s\n' % (label, arr, cipher))
os.close(fd)
