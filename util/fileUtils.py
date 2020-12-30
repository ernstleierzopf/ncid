import os


def write_ciphertexts_with_keys_to_file(filename, ciphertexts, keys):
    for i in range(0, len(ciphertexts)):
        ciphertexts[i] += ',' + keys[i]
    write_txt_list_to_file(filename, ciphertexts)


def write_txt_list_to_file(filename, texts):
    with open(filename, 'wb') as file:
        for line in texts:
            if isinstance(line, str):
                line = line.encode()
            file.write(line)
            file.write(b'\n')


def read_txt_list_from_file(filename):
    with open(filename, 'rb') as file:
        return file.readlines()


def unpack_zip_folders(path):
    import zipfile
    if not os.path.isdir(path):
        return
    dir_name = os.listdir(path)
    for name in dir_name:
        if name.lower().endswith('.zip') and '-' not in name:
            full_path = os.path.join(path, name)
            if os.path.exists(full_path.split('.zip')[0]):
                os.remove(full_path.split('.zip')[0])
            with zipfile.ZipFile(full_path, 'r') as zip_ref:
                print("extracting %s" % name)
                zip_ref.extractall(path)
            if os.path.exists(full_path):
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


def print_progress(output_str, file_counter, total_file_count, factor=100):
    # console output for % of read files
    if file_counter % factor == 0 or file_counter == total_file_count:
        percentage = int(float(file_counter) / float(total_file_count) * 100)
        output = output_str
        for _ in range(0, int(percentage * 0.2)):
            output += '#'
        for _ in range(int(percentage * 0.2), 20):
            output += '.'
        output = output + '] ' + str(file_counter) + '/' + str(total_file_count) + ' (' + str(percentage) + "%)"
        print(output)
