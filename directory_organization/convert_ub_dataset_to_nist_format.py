from fileProcessingUtil import *
import os
from PIL import Image
import shutil
import json

# passed
def testRidgebaseRenaming():
    filenames = ['1_10727_Left_Index.bmp', '1_11826_Left_Little.bmp', \
        '1_12385_Left_Middle.bmp', '1_12407_Left_Ring.bmp', \
        '1_12536_Right_Index.bmp', '1_14366_Right_Little.bmp', \
        '1_15445_Right_Middle.bmp', '2_74850_Right_Ring.bmp', \
        '1_Apple_10727_1_LEFT_image_fingerprint8GIFPDNU_0.9817909002304077_0.png', \
        '1_google_54385_2_LEFT_image_fingerprintNF7PTOEL_0.979097843170166_1.png', \
        '1_Apple_10727_1_LEFT_image_fingerprint8GIFPDNU_0.9961502552032471_2.png', \
        '2_google_28327_1_LEFT_image_fingerprint2XICSA9S_0.9721018671989441_3.png', \
        '2_google_85977_3_RIGHT_image_fingerprint05LCHDLP_0.9929990768432617_0.png', \
        '1_Apple_10727_1_RIGHT_image_fingerprintDAH28AVI_0.8899842500686646_1.png', \
        '1_google_65030_1_RIGHT_image_fingerprintUKND42XV_2.png', \
        '1_Apple_10727_1_RIGHT_image_fingerprintDAH28AVI_0.9721535444259644_3.png', \
        ]
    for the_filename in filenames:
        print('original filename:', the_filename)
        print('\trenamed filename:', rename_ridgebase_file(the_filename))
    return

RIDGEBASE_ORIGINAL_TRAIN_DIR = '/data/therealgabeguo/fingerprint_data/RidgeBase_UB_Dataset/Task1/Train'
RIDGEBASE_NEW_DIR = '/data/therealgabeguo/fingerprint_data/RidgeBase_Processed'

if __name__ == "__main__":
    os.makedirs(RIDGEBASE_NEW_DIR, exist_ok=True)

    root_dir = RIDGEBASE_ORIGINAL_TRAIN_DIR
    old2new = dict()
    new2old = dict()

    for (dirpath, dirnames, filenames) in os.walk(root_dir):
        for the_filename in filenames:
            # only get images
            if the_filename[-4:].lower() not in ['.png', '.jpg', '.bmp']:
                print('not an image:', the_filename)
                continue
            # create filepaths
            src = os.path.join(dirpath, the_filename)
            new_filepath = os.path.join(RIDGEBASE_NEW_DIR, rename_ridgebase_file(the_filename))
 
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

            # can copy image if is sensor image, otherwise, need to rotate
            if is_ub_contactbased_sample(the_filename):
                #print('copying from {} to {}'.format(src, new_filepath))
                shutil.copyfile(src, new_filepath)
            else:
                # left fingers are pointing right, so need to rotate 90 degrees CC
                if 'left' in the_filename.lower():
                    rotation = 90
                # right fingers are pointing left, so need to rotate 90 degrees CW (270 CC)
                elif 'right' in the_filename.lower():
                    rotation = 270
                else:
                    raise ValueError('invalid filename: {}'.format(the_filename))
                #print('copying and rotating from {} to {}'.format(src, new_filepath))
                image = Image.open(src)
                image = image.rotate(rotation, resample=Image.NEAREST, expand=True)
                image.save(new_filepath)

    log_path = os.path.join(RIDGEBASE_NEW_DIR, 'old2new_fileMappings.json')
    with open(log_path, 'w') as fout:
        json.dump(old2new, fout, indent=4, sort_keys=True)

    print(len(old2new))
    # end