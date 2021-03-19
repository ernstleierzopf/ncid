import os

ciphers = ['amsco', 'bazeries', 'beaufort', 'bifid', 'cmbifid', 'digrafid', 'foursquare', 'fractionated_morse', 'gromark', 'gronsfeld',
           'homophonic', 'monome_dinome', 'morbit', 'myszkowski', 'nicodemus', 'nihilist_substitution', 'periodic_gromark',
           'phillips', 'playfair', 'pollux', 'porta', 'portax', 'progressive_key', 'quagmire2', 'quagmire3', 'quagmire4', 'ragbaby',
           'redefence', 'seriated_playfair', 'slidefair', 'swagman', 'tridigital', 'trifid', 'tri_square', 'two_square', 'vigenere']
src_dir = '../data/aca_predictions100'
dest_file = '../data/aca_predictions_undefined_count100.txt'
dest_fd = os.open(dest_file, os.O_WRONLY | os.O_CREAT)
for file in os.listdir(src_dir):
    count = 0
    with open(os.path.join(src_dir, file), 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line == '' or 'length:' in line:
                continue
            cipher = line.split(' ')[0]
            if cipher not in ciphers:
                count += 1
        os.write(dest_fd, b"%s: %d\n" % (file.encode(), count))
os.close(dest_fd)
