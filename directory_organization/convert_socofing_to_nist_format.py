from fileProcessingUtil import *
import os
import shutil
import json

def testSocofingRename():
    filenames = [('Real', '1__M_Left_index_finger.BMP'), ('Real', '1__M_Left_little_finger.BMP'),
    ('Altered-Easy', '2__F_Left_middle_finger_CR.BMP'), ('Altered-Easy', '2__F_Left_ring_finger_Obl.BMP'),
    ('Altered-Medium', '3__M_Left_thumb_finger_Zcut.BMP'), ('Altered-Medium', '4__M_Right_index_finger_CR.BMP'),
    ('Altered-Hard', '56__F_Right_little_finger_Obl.BMP'), ('Altered-Hard', '345__M_Right_middle_finger_CR.BMP'),
    ('Real', '436__M_Right_ring_finger.BMP'), ('Altered-Medium', '11__M_Right_thumb_finger_Zcut.BMP')]

    for the_filename in filenames:
        print('Orig:', the_filename)
        print('\tRenamed:', rename_socofing_file(the_filename[1], the_filename[0]))
    
    return

SOCOFING_ORIGINAL_DIR = '/data/therealgabeguo/fingerprint_data/SOCOFing'
SOCOFING_NEW_DIR = '/data/therealgabeguo/fingerprint_data/SOCOFing_Renamed'

if __name__ == "__main__":
    os.makedirs(SOCOFING_NEW_DIR, exist_ok=True)

    root_dir = SOCOFING_ORIGINAL_DIR
    old2new = dict()
    new2old = dict()

    for (dirpath, dirnames, filenames) in os.walk(root_dir):
        alterationDifficulty = dirpath.split('/')[-1]
        if len(filenames) > 0:
            assert alterationDifficulty in ['Real', 'Altered-Easy', 'Altered-Medium', 'Altered-Hard']
        
        for the_filename in filenames:
            # only get images
            if the_filename[-4:].lower() not in ['.png', '.jpg', '.bmp']:
                print('not an image:', the_filename)
                continue
            # create filepaths
            src = os.path.join(dirpath, the_filename)
            new_filepath = os.path.join(SOCOFING_NEW_DIR, rename_socofing_file(the_filename, alterationDifficulty))
 
            if new_filepath in old2new.values():
                #print('duplicate: {}'.format(new_filepath))
                print('duplicate: {}'.format(new_filepath))
                print('\tsource: {}'.format(new2old[new_filepath]))
                print('\tsource: {}'.format(src))

            old2new[src] = new_filepath
            new2old[new_filepath] = src

            if os.path.exists(new_filepath):
                print('already exists:', new_filepath)
                continue
            
            #print(src, '\n\t', new_filepath)
            shutil.copyfile(src, new_filepath)

    log_path = os.path.join(SOCOFING_NEW_DIR, 'old2new_fileMappings.json')
    with open(log_path, 'w') as fout:
        json.dump(old2new, fout, indent=4, sort_keys=True)

    print(len(old2new))
    # end