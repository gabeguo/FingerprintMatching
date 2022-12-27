import os, shutil
for file in os.listdir('test_input'):
    if '.png' in file:
        filepath = os.path.join('test_input', file)
        pid = file.split('_')[0]
        dest = os.path.join('test_input', pid)
        os.makedirs(dest, exist_ok=True)
        shutil.move(filepath, dest)
        #print(file)
