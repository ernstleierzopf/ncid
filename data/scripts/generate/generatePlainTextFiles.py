import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='CANN Plaintext Generator Script')
parser.add_argument('--input_folder', default='../../gutenberg_en', type=str,
                    help='Input folder of the plaintexts to be extracted.')
parser.add_argument('--save_folder', default='../../plaintexts',
                    help='Directory for saving extracted plaintexts.')
parser.add_argument('--save_file_size', default=10, type=int,
                    help='The number of Megabytes written before starting a new file.'\
                         'The saved files are numbered, starting with 1.')
parser.add_argument('--save_file_name_base', default='plaintexts', type=str,
                    help='Base name of the output files. The files are also numbered '\
                         'depending on the --save_file_size parameter.')
parser.add_argument('--keep_spaces', default=False, type=str2bool,
                    help='Keep spaces in the plaintext files.')
parser.add_argument('--sentences_count', default=50000, type=int,
                    help='The number of random sentences to be extracted.')
parser.add_argument('--min_sentence_length', default=50, type=int,
                    help='Defines the minimal number of characters in a sentence to be chosen.')
parser.add_argument('--max_sentence_length', default=150, type=int,
                    help='Defines the maximal number of characters in a sentence to be chosen.')


args = parser.parse_args()
parser.set_defaults(keep_spaces=False)

#print all arguments for debugging..
for arg in vars(args):
    print("%s = %s"%(arg, vars(args)[arg]))