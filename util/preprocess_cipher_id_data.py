lines = []
with open("../data/original_cipher-id-ciphertext.txt", 'r') as fd:
    for line in fd:
        if line == "\n":
            continue
        lines.append(line.split(" ")[1])

with open("../data/cipher-id-ciphertext.txt", "w") as fd:
    fd.writelines(lines)
