import os

FOLDER_STRUCTURE = ('gutenberg', 'common')

def write_ciphertexts_with_keys_to_file(filename, ciphertexts, keys, cipherImplementation) :
    with open(filename,'wb') as file :
        for i in range(0,len(ciphertexts)) :
            if isinstance(ciphertexts[i], str):
                ciphertexts[i] = ciphertexts[i].encode()
            file.write(ciphertexts[i] + ',' + cipherImplementation.mapNumbersIntoTextspace(keys[i]))
            file.write('\n')

def write_txt_list_to_file(filename, texts) :
    with open(filename,'wb') as file :
        for line in texts :
            if isinstance(line, str):
                line = line.encode()
            file.write(line)
            file.write('\n')

def read_txt_list_from_file(filename):
    with open(filename,'rb') as file:
        return file.readlines()

'''
This function searches recursively for all textfiles stored in the path and its sub-folders
It stores the found filenames (full paths) in cache_file and returns the list of filenames.
'''
def find_textfiles(path, folder_structure, restructure_folder_flag) :
    file_counter = 0
    txt_files = []
    if restructure_folder_flag:
        unpack_zip_folders(path)
    dir = os.listdir(path)
    for name in dir:
        if os.path.isdir(os.path.join(path, name)) :
            txt_files += find_textfiles(os.path.join(path, name), folder_structure, restructure_folder_flag)
        elif name.lower().endswith('.txt') and '-' not in name:
            if os.path.join(path, name) not in txt_files:
                txt_files.append(os.path.join(path, name))
                remove_disclaimer_from_file(os.path.join(path, name))
                file_counter = file_counter + 1
            # console output for % of read files
            if file_counter % 10 == 0 :
                percentage = int( float(file_counter) / float(len(dir)) * 100)
                output = 'Collecting files: ['
                for i in range(0, int(percentage * 0.2)) :
                    output += '#'
                for i in range( int(percentage * 0.2), 20) :
                    output += '.'
                output = output + '] ' + str(file_counter) + ' from ' + str(path) + ' ('  + str(percentage) + "%)"
                print(output)
        elif restructure_folder_flag:
            if os.path.exists(os.path.join(path, name)):
                os.remove(os.path.join(path, name))
    if os.path.exists(path) and len(os.listdir(path)) == 0:
        os.removedirs(path)
    return txt_files

def restructure_folder_and_write_caches(restructure_folder_flag, txt_files, path):
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

def unpack_zip_folders(path):
    import zipfile
    dir = os.listdir(path)
    for name in dir:
        if name.lower().endswith('.zip') and '-' not in name:
            with zipfile.ZipFile(os.path.join(path, name), 'r') as zip_ref:
                zip_ref.extractall(path)
            if os.path.exists(os.path.join(path, name)):
                os.remove(os.path.join(path, name))

def remove_disclaimer_from_file(file):
    with open(file, 'rb') as f:
        txt = f.read()
    old = txt
    txt = txt.replace(b'\r\n', b'\n')
    while txt.find(b'\n\n\n') != -1:
        txt = txt.replace(b'\n\n\n', b'\n\n')
    pos = txt.find(b'*** START OF THIS PROJECT')
    if pos < 0:
        pos = txt.find(b'***START OF THIS PROJECT')
    if pos > -1:
        txt = txt[pos:]
        pos = txt.find(b'\n')
        txt = txt[pos:]
    while txt.find(b'\n') == 0:
        txt = txt[1:]

    if old != txt:
        with open(file, 'wb') as f:
            f.write(txt)