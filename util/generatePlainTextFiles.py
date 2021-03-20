import argparse
import os
from utils import unpack_zip_folders, remove_disclaimer_from_file, print_progress


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def find_textfiles(path, restructure_folder_flag, total_fc):
    """This function searches recursively for all textfiles stored in the path and its sub-folders. It stores the found filenames
    (full paths) in cache_file and returns the list of filenames."""
    file_counter = 0
    found_files = []
    if restructure_folder_flag:
        unpack_zip_folders(path)
    dir_name = os.listdir(path)
    for name in dir_name:
        if os.path.isdir(os.path.join(path, name)):
            file_counter += 1
            found_files += find_textfiles(os.path.join(path, name), restructure_folder_flag, total_fc)
        elif name.lower().endswith('.txt') and '-' not in name and 'robots.txt' not in name and os.path.join(path, name) not in found_files:
            found_files.append(os.path.join(path, name))
            remove_disclaimer_from_file(os.path.join(path, name))
            file_counter += 1
        elif restructure_folder_flag:
            if os.path.exists(os.path.join(path, name)):
                os.remove(os.path.join(path, name))
                total_fc -= 1
        print_progress('Collecting files: [', file_counter, total_fc)
    if os.path.exists(path) and len(os.listdir(path)) == 0 and restructure_folder_flag:
        os.removedirs(path)
    return found_files


def restructure_folder(restructure_folder_flag, found_files, path):
    if restructure_folder_flag:
        restructured_txt_files = []
        for file in found_files:
            restructured_path = os.path.join(path, os.path.basename(file))
            if restructured_path != file and os.path.exists(restructured_path):
                print('Path \'%s\' already exists!' % restructured_path)
            if file != restructured_path:
                os.rename(file, restructured_path)
            print_progress('Restructuring files: [', found_files.index(file), total_file_count)
            restructured_txt_files.append(restructured_path)
            dir_name = os.path.dirname(file)
            while len(os.listdir(dir_name)) == 0:
                os.removedirs(os.path.dirname(file))
                dir_name = os.path.dirname(dir_name)
        found_files = restructured_txt_files
    return found_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CANN Plaintext Generator Script')
    parser.add_argument('--directory', default='../gutenberg_en', type=str,
                        help='Input and save folder of the plaintexts to be extracted.')
    parser.add_argument('--restructure_directory', default=False, type=str2bool,
                        help='Deletes all unneeded files from the --directory recursively and moves the files into the first subdirectory.'
                             'Be careful! This option removes files and directories. Only use it with a backup of the files!')
    args = parser.parse_args()
    args.directory = os.path.abspath(args.directory)

    # print all arguments for debugging..
    for arg in vars(args):
        print("%s = %s" % (arg, vars(args)[arg]))

    total_file_count = 0
    for root, dirs, files in os.walk(args.directory):
        total_file_count += len(files)

    txt_files = find_textfiles(args.directory, args.restructure_directory, total_file_count)
    txt_files = restructure_folder(args.restructure_directory, txt_files, args.directory)
