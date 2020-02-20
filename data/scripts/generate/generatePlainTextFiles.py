import argparse
import os
import sys

sys.path.append("../../../")
from utilities import fileUtils

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='CANN Plaintext Generator Script')
parser.add_argument('--input_folder', default='../../gutenberg_test', type=str,
                    help='Input folder of the plaintexts to be extracted.')
parser.add_argument('--restructure_input_folder', default=False, type=str2bool,
                    help='Deletes all unneeded files from the input_folder recursively '\
                         'and moves the files into the first subdirectory.'\
                         'This option can only be used with input_folder_structure=\'gutenberg\'.')
parser.add_argument('--input_folder_structure', default=fileUtils.FOLDER_STRUCTURE[0],
                    choices=fileUtils.FOLDER_STRUCTURE, help='Sets the structure of the input folder.'\
                    'This is needed to be able to handle different directory structures and not to'\
                    'reuse the same file in another format. Possible values are \'gutenberg\' for the'\
                    'gutenberg-library or common, if you want to use all files in the subdirectories.')
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
parser.add_argument('--sentences_count', default=None, type=int,
                    help='The number of random sentences to be extracted.'\
                    'If None, all sentences are extracted.')
parser.add_argument('--min_sentence_length', default=None, type=int,
                    help='Defines the minimal number of characters in a sentence to be chosen.'
                    'If this parameter is None, no minimal sentence length will be checked.')
parser.add_argument('--max_sentence_length', default=None, type=int,
                    help='Defines the maximal number of characters in a sentence to be chosen.'\
                    'If this parameter is None, no maximal sentence length will be checked.')

args = parser.parse_args()

args.input_folder = os.path.abspath(args.input_folder)
args.save_folder = os.path.abspath(args.save_folder)

if args.restructure_input_folder and args.input_folder_structure != fileUtils.FOLDER_STRUCTURE[0]:
    print("The parameter --restructure_input_folder can only be used with --input_folder_structure=%s"%fileUtils.FOLDER_STRUCTURE[0], file=sys.stderr)

#print all arguments for debugging..
for arg in vars(args):
    print("%s = %s"%(arg, vars(args)[arg]))

else:
    txt_files = fileUtils.find_textfiles(args.input_folder, args.input_folder_structure, args.restructure_input_folder)
    txt_files = fileUtils.restructure_folder_and_write_caches(args.restructure_input_folder, txt_files, args.input_folder)

