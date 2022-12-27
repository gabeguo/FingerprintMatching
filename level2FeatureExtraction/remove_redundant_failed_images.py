with open('failed_images.txt', 'r') as fin:
    lines = set([line.strip() for line in fin.readlines() if line.strip() != ''])
with open('failed_images.txt', 'w') as fout:
    for line in lines:
        fout.write(line)
        fout.write('\n')
