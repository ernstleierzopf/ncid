from cipherImplementations.simpleSubstitution import SimpleSubstitution
from cipherImplementations.vigenere import Vigenere
from cipherImplementations.columnarTransposition import ColumnarTransposition
from cipherImplementations.playfair import Playfair
from cipherImplementations.hill import Hill
from cipherImplementations.amsco import Amsco
from cipherImplementations.autokey import Autokey
from cipherImplementations.baconian import Baconian
from cipherImplementations.bazeries import Bazeries
from cipherImplementations.beaufort import Beaufort
from cipherImplementations.bifid import Bifid
from cipherImplementations.cadenus import Cadenus
from cipherImplementations.checkerboard import Checkerboard
from cipherImplementations.condi import Condi
from cipherImplementations.cmbifid import CMBifid

ALPHABET = b'abcdefghijklmnopqrstuvwxyz'
UNKNOWN_SYMBOL = b'?'
UNKNOWN_SYMBOL_NUMBER = 90

# CIPHER_TYPES = ['columnar_transposition', 'hill', 'playfair', 'simple_substitution', 'vigenere']
CIPHER_TYPES = ['amsco', 'autokey', 'baconian', 'bazeries', 'beaufort', 'bifid', 'cadenus', 'checkboard', 'columnar_transposition', 'condi',
                'cmbifid']
# CIPHER_IMPLEMENTATIONS = [ColumnarTransposition(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
#                           Hill(ALPHABET, b'x', ord('x')),
#                           Playfair(ALPHABET.replace(b'j', b''), b'x', ord('x')),
#                           SimpleSubstitution(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
#                           Vigenere(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER)]
CIPHER_IMPLEMENTATIONS = [Amsco(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Autokey(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Baconian(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Bazeries(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Beaufort(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Bifid(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Cadenus(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Checkerboard(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          ColumnarTransposition(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Condi(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          CMBifid(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER)]

# KEY_LENGTHS = [13, None, 13, None, 13]
# KEY_LENGTHS = [list(range(4, 17)), [None]*13, list(range(4, 17)), [None]*13, list(range(4, 17))]
# KEY_LENGTHS = [[5,10,20,25], [None]*4, [6,7,8,9], [None]*4, [5,10,20,25]]
# KEY_LENGTHS = [[None]*4, [5,10,20,25]]
KEY_LENGTHS = [[5,6,7,8], [5,6,7,8], [None]*4, [None]*4, [5,6,7,8], [5,6,7,8], [4,4,4,4], [5,10,15,20], [5,6,7,8], [None]*4, [5,6,7,8]]
MTC3 = 'mtc3'
ACA = 'aca'