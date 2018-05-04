import os

in_folder = './data/wikipedia'
out_folder = './data/wikipedia_filtered'
flist = os.listdir(in_folder)
current_index = 0
for f in flist:
    print(f)
    lines = []
    with open(os.path.join(in_folder, f), "r", encoding="ISO-8859-1") as f_text:
        for l in f_text:
            if not l.startswith('<doc id=') and \
               not l.startswith('ENDOFARTICLE.') and \
               not l.startswith('</doc>') and \
               not l.startswith('\n'):
                lines.append(l)

    with open(os.path.join(out_folder, f), "w", encoding="utf8") as f_out_text:
        f_out_text.writelines(lines)

