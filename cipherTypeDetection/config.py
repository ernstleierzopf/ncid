from cipherImplementations.simpleSubstitution import SimpleSubstitution
from cipherImplementations.vigenere import Vigenere
from cipherImplementations.hill import Hill
from cipherImplementations.amsco import Amsco
from cipherImplementations.autokey import Autokey
from cipherImplementations.baconian import Baconian
from cipherImplementations.bazeries import Bazeries
from cipherImplementations.beaufort import Beaufort
from cipherImplementations.bifid import Bifid
from cipherImplementations.cadenus import Cadenus
from cipherImplementations.checkerboard import Checkerboard
from cipherImplementations.columnarTransposition import ColumnarTransposition
from cipherImplementations.condi import Condi
from cipherImplementations.cmbifid import CMBifid
from cipherImplementations.digrafid import Digrafid
from cipherImplementations.foursquare import Foursquare
from cipherImplementations.fractionedMorse import FractionedMorse
from cipherImplementations.grandpre import Grandpre
from cipherImplementations.grille import Grille
from cipherImplementations.gromark import Gromark
from cipherImplementations.gronsfeld import Gronsfeld
from cipherImplementations.headlines import Headlines
from cipherImplementations.homophonic import Homophonic
from cipherImplementations.keyPhrase import KeyPhrase
from cipherImplementations.monomeDinome import MonomeDinome
from cipherImplementations.morbit import Morbit
from cipherImplementations.myszkowski import Myszkowski
from cipherImplementations.nicodemus import Nicodemus
from cipherImplementations.nihilistTransposition import NihilistTransposition
from cipherImplementations.null import Null
from cipherImplementations.numberedKey import NumberedKey
from cipherImplementations.periodicGromark import PeriodicGromark
from cipherImplementations.phillips import Phillips
from cipherImplementations.phillipsRC import PhillipsRC
from cipherImplementations.plaintext import Plaintext
from cipherImplementations.playfair import Playfair
from cipherImplementations.pollux import Pollux
from cipherImplementations.porta import Porta
from cipherImplementations.portax import Portax
from cipherImplementations.progressiveKey import ProgressiveKey
from cipherImplementations.quagmire import Quagmire

ALPHABET = b'abcdefghijklmnopqrstuvwxyz'
UNKNOWN_SYMBOL = b'?'
UNKNOWN_SYMBOL_NUMBER = 90

# CIPHER_TYPES = ['columnar_transposition', 'hill', 'playfair', 'simple_substitution', 'vigenere']
CIPHER_TYPES = ['amsco', 'autokey', 'baconian', 'bazeries', 'beaufort', 'bifid', 'cadenus', 'checkboard', 'columnar_transposition', 'condi',
                'cmbifid', 'digrafid', 'foursquare', 'fractioned_morse', 'grandpre', 'grille', 'gromark', 'gronsfeld', 'headlines',
                'key_phrase', 'homophonic',  # , 'incomplete_columnar_transposition'
                'monome_dinome', 'morbit', 'myszkowski', 'nicodemus', 'nihilist_transposition', 'null', 'numbered_key', 'periodic_gromark',
                'phillips', 'phillips_rc', 'plaintext', 'playfair', 'pollux', 'porta', 'portax', 'progressive_key', 'quagmire1',
                'quagmire2', 'quagmire3', 'quagmire4']
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
                          ColumnarTransposition(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER, fill_blocks=True),
                          Condi(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          CMBifid(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Digrafid(ALPHABET + b'#', UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Foursquare(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          FractionedMorse(ALPHABET + b' ', UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Grandpre(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Grille(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Gromark(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Gronsfeld(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Headlines(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Homophonic(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          # incomplete Columnar Transposition: how is it differentiable to the normal columnar transposition?
                          # ColumnarTransposition(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER, fill_blocks=True)
                          KeyPhrase(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          MonomeDinome(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Morbit(ALPHABET + b' ', UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Myszkowski(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Nicodemus(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          NihilistTransposition(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Null(ALPHABET + b' ', UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          NumberedKey(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          PeriodicGromark(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Phillips(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          PhillipsRC(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Plaintext(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Playfair(ALPHABET.replace(b'j', b''), UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Pollux(ALPHABET + b' ', UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Porta(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Portax(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          ProgressiveKey(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER),
                          Quagmire(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER, keyword_type=1),
                          Quagmire(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER, keyword_type=2),
                          Quagmire(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER, keyword_type=3),
                          Quagmire(ALPHABET, UNKNOWN_SYMBOL, UNKNOWN_SYMBOL_NUMBER, keyword_type=4)]

# KEY_LENGTHS = [13, None, 13, None, 13]
# KEY_LENGTHS = [list(range(4, 17)), [None]*13, list(range(4, 17)), [None]*13, list(range(4, 17))]
# KEY_LENGTHS = [[5,10,20,25], [None]*4, [6,7,8,9], [None]*4, [5,10,20,25]]
# KEY_LENGTHS = [[None]*4, [5,10,20,25]]
KEY_LENGTHS = [[5,6,7,8], [5,6,7,8], [None]*4, [None]*4, [5,6,7,8], [5,6,7,8], [4,4,4,4], [5,10,15,20], [5,6,7,8], [5,6,7,8], [5,6,7,8],
               [5,6,7,8], [5,6,7,8], [5,6,7,8], [None]*4, [2,5,10,5], [5,6,7,8], [5,6,7,8], [5,6,7,8], [None]*4, [None]*4, [None]*4,
               [None]*4, [5,6,7,8], [5,6,7,8], [10,10,10,10], [None]*4, [5,6,7,8], [5,6,7,8], [5,6,7,8], [5,6,7,8], [None]*4, [5,6,7,8],
               [None]*4, [5,6,7,8], [5,6,7,8], [5,6,7,8], [5,6,7,8], [5,6,7,8], [5,6,7,8], [5,6,7,8]]
MTC3 = 'mtc3'
ACA = 'aca'