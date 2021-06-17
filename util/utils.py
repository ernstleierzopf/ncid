import numpy as np
import os


def map_text_into_numberspace(text, alphabet, unknown_symbol_number):
    array = []
    for c in text:
        if bytes([c]) in alphabet:
            array.append(alphabet.index(bytes([c])))
        else:
            array.append(unknown_symbol_number)
    return np.array(array)


def map_numbers_into_textspace(numbers, alphabet, unknown_symbol):
    output = b''
    for n in numbers:
        if n >= len(alphabet):
            output = output + unknown_symbol
        else:
            output = output + bytes([alphabet[n]])
    return output


def remove_unknown_symbols(text, alphabet):
    i = 0
    while i < len(text):
        if text[i] not in alphabet:
            text = text.replace(bytes([text[i]]), b'')
        else:
            i += 1
    return text


# morse code in alphabetical order
morse_codes = ['.-', '-...', '-.-.', '-..', '.', '..-.', '--.', '....', '..', '.---', '-.-', '.-..', '--', '-.', '---', '.--.',
               '--.-', '.-.', '...', '-', '..-', '...-', '.--', '-..-', '-.--', '--..']


def encrypt_morse(plaintext):
    morse_code = ''
    for c in plaintext:
        if c == 26:
            morse_code += 'x'
            continue
        morse_code += morse_codes[c] + 'x'
    morse_code += 'x'
    return morse_code


def decrypt_morse(ciphertext, key_morse, key):
    morse_code = ''
    for c in ciphertext:
        morse_code += key_morse[np.where(key == c)[0][0]]
    morse_code += 'x'
    return morse_code


def get_model_input_length(model_, arch):
    input_length = None
    if arch == "LSTM":
        input_length = model_.layers[0].input_length
    elif arch == "CNN":
        input_length = model_.layers[0].input_shape[1]
    elif arch == "Transformer":
        input_length = model_.layers[0].input_shape[0][1]
    elif arch == "Ensemble":
        for i in range(len(model_.architectures)):
            if model_.architectures[i] in ("LSTM", "CNN", "Transformer"):
                return get_model_input_length(model_.models[i], model_.architectures[i])
        return None

    return input_length


def write_ciphertexts_with_keys_to_file(filename, ciphertexts, keys):
    for i in range(0, len(ciphertexts)):
        ciphertexts[i] += ',' + keys[i]
    write_txt_list_to_file(filename, ciphertexts)


def write_txt_list_to_file(filename, texts):
    with open(filename, 'wb') as file:
        for line in texts:
            if isinstance(line, str):
                line = line.encode()
            if not isinstance(line, bytes):
                line = str(line).encode()
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
