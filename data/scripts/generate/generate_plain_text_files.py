import argparse
import os
import sys

sys.path.append("../../../")
from util import file_utils

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

'''
This function searches recursively for all textfiles stored in the path and its sub-folders
It stores the found filenames (full paths) in cache_file and returns the list of filenames.
'''
def find_textfiles(path, restructure_folder_flag, total_file_count) :
    file_counter = 0
    txt_files = []
    if restructure_folder_flag:
        file_utils.unpack_zip_folders(path)
    dir = os.listdir(path)
    for name in dir:
        if os.path.isdir(os.path.join(path, name)) :
            file_counter += 1
            txt_files += find_textfiles(os.path.join(path, name), restructure_folder_flag, total_file_count)
        elif name.lower().endswith('.txt') and '-' not in name and 'robots.txt' not in name:
            if os.path.join(path, name) not in txt_files:
                txt_files.append(os.path.join(path, name))
                file_utils.remove_disclaimer_from_file(os.path.join(path, name))
                file_counter += 1
        elif restructure_folder_flag:
            if os.path.exists(os.path.join(path, name)):
                os.remove(os.path.join(path, name))
                total_file_count -= 1
        file_utils.print_progress('Collecting files: [', file_counter, total_file_count, name)
    if os.path.exists(path) and len(os.listdir(path)) == 0 and restructure_folder_flag:
        os.removedirs(path)
    return txt_files

def restructure_folder(restructure_folder_flag, txt_files, path):
    if restructure_folder_flag:
        restructured_txt_files = []
        for file in txt_files:
            restructured_path = os.path.join(path, os.path.basename(file))
            if restructured_path != file and os.path.exists(restructured_path):
                print('Path \'%s\' already exists!'%restructured_path)
            os.rename(file, restructured_path)
            restructured_txt_files.append(restructured_path)
            dir = os.path.dirname(file)
            while len(os.listdir(dir)) == 0:
                os.removedirs(os.path.dirname(file))
                dir = os.path.dirname(dir)
        txt_files = restructured_txt_files
    return txt_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CANN Plaintext Generator Script')
    parser.add_argument('--directory', default='../../gutenberg_test', type=str,
                        help='Input and save folder of the plaintexts to be extracted.')
    parser.add_argument('--restructure_directory', default=False, type=str2bool,
                        help='Deletes all unneeded files from the --directory recursively '\
                             'and moves the files into the first subdirectory.'\
                             'Be careful! This option removes files and directories.'\
                             'Only use it with a backup of the files!')
    args = parser.parse_args()
    args.directory = os.path.abspath(args.directory)

    #print all arguments for debugging..
    for arg in vars(args):
        print("%s = %s"%(arg, vars(args)[arg]))

    total_file_count = 0
    for root, dirs, files in os.walk(args.directory):
        total_file_count += len(files)

    txt_files = find_textfiles(args.directory, args.restructure_directory, total_file_count)
    txt_files = restructure_folder(args.restructure_directory, txt_files, args.directory)

