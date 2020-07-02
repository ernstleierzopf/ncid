lines = []
with open("../original_cipher-id-ciphertext.txt", 'r') as fd:
    for line in fd:
        if line == "\n":
            continue
        lines.append(line.split(" ")[1])

with open("../cipher-id-ciphertext.txt", "w") as fd:
    fd.writelines(lines)