USE_DICTIONARY = False
GENERATE_RANDOM_ALPHABETS = False
WORD_DICT = {}
UNIQUE_WORD_DICT = {}

if USE_DICTIONARY:
    with open('../data/word_lists/wiki-270k.txt', 'rb') as f:
        words = [line.rstrip(b'\n') for line in f]
    for word in words:
        if len(word) not in WORD_DICT:
            WORD_DICT[len(word)] = []
        WORD_DICT[len(word)].append(word)
        if len(word) == len(set(word)):
            if len(word) not in UNIQUE_WORD_DICT:
                UNIQUE_WORD_DICT[len(word)] = []
            UNIQUE_WORD_DICT[len(word)].append(word)
