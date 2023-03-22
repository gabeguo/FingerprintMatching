"""

This script copies the data
    into a duplicate of the data directory, except it excludes
    person-sensor combos for which all 10 fingerprints are not available.
This makes the dataset balanced.

It assumes that the original data directory was created via
    split_people.py

Start State:
    data_folder
        /train
            /00002650
                00002650_S_500_slap_01.png
                00002650_S_500_slap_02.png
                00002650_S_500_slap_03.png
                00002650_S_500_slap_04.png
                00002650_S_500_slap_05.png
                00002650_S_500_slap_06.png
                00002650_S_500_slap_07.png
                00002650_S_500_slap_08.png
                00002650_S_500_slap_09.png
                00002650_S_500_slap_10.png
        /val
            /00002651
                00002651_S_500_slap_01.png
                00002651_S_500_slap_02.png
                00002651_S_500_slap_03.png
                00002651_S_500_slap_05.png
                00002651_S_500_slap_06.png
                00002651_S_500_slap_08.png
                00002651_S_500_slap_09.png
                00002651_S_500_slap_10.png
        /test
            /00002646
                00002646_S_500_slap_05.png
                00002646_K_500_slap_04.png

End State:
    data_folder_subset
        /train
            /00002650
                00002650_S_500_slap_01.png
                00002650_S_500_slap_02.png
                00002650_S_500_slap_03.png
                00002650_S_500_slap_04.png
                00002650_S_500_slap_05.png
                00002650_S_500_slap_06.png
                00002650_S_500_slap_07.png
                00002650_S_500_slap_08.png
                00002650_S_500_slap_09.png
                00002650_S_500_slap_10.png
        /val
            /00002651
        /test
            /00002646

"""

import sys, getopt, os, shutil
from fileProcessingUtil import *

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

def create_train_val_test_subfolders(data_folder):
    for subfolder in [TRAIN, VAL, TEST]:
        subdirectory_abs_path = os.path.join(data_folder, subfolder)
        if not os.path.exists(subdirectory_abs_path):
            os.mkdir(subdirectory_abs_path)
    return

# Checks if for this person-sensor combo, do we have all 10 fingerprints?
# Ex: Given: 00002650_S_500_slap_01.png; do we have _02, _03, etc.?
def all_10_samples_exist(orig_path, all_sensors_need_10_samples):
    fgrps = ['_01', '_02', '_03', '_04', '_05', '_06', '_07', '_08', '_09', '_10']
    fgrp_index = orig_path[:orig_path.rfind("_")].rfind("_") if 'MOD' in orig_path else orig_path.rfind('_')
    for fgrp in fgrps:
        found_valid_path = False
        for the_mod in ['', '_MOD11', '_MOD12']:
            alt_finger = orig_path[:fgrp_index] + the_mod + fgrp + orig_path[orig_path.rfind('.'):]
            if os.path.exists(alt_finger):
                found_valid_path = True
                break

        if not found_valid_path:
            return False
    
    if all_sensors_need_10_samples:
        person_folder = os.path.dirname(orig_path)
        if len(os.listdir(person_folder)) % 10 != 0:
            return False
        
    return True

"""
Copies the files from data_folder into balanced_data_folder that are from desired_sensors
"""
def copy_files(data_folder, balanced_data_folder, all_sensors_need_10_samples):
    create_train_val_test_subfolders(balanced_data_folder)

    # https://stackoverflow.com/questions/1192978/python-get-relative-path-of-all-files-and-subfolders-in-a-directory
    for dirpath, dirnames, filenames in os.walk(data_folder):
        for filename in filenames:
            orig_path = os.path.join(dirpath, filename)
            if any(ending in filename for ending in ['.jpg', '.png', '.jpeg', '.pneg']) \
                    and all_10_samples_exist(orig_path, all_sensors_need_10_samples): # make sure we have all 10 samples from this person
                rel_dir = os.path.relpath(dirpath, data_folder)
                rel_file = os.path.join(rel_dir, filename)
                
                new_path = os.path.join(balanced_data_folder, rel_file)
                new_dir = os.path.join(balanced_data_folder, rel_dir)
                
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)

                #print('\t', orig_path, '\n\t', new_path, '\n')
                shutil.copy(orig_path, new_path)
    return

def main():
    # Variables we need
    data_folder = None
    all_sensors_need_10_samples = False

    argv = sys.argv[1:] 

    # help message
    usage_msg = "Usage: get_all_10_finger_samples.py --data_folder <directory address> [--all_sensors_need_10_samples]"
    
    try:
        opts, args = getopt.getopt(argv, "h", ['help', 'data_folder=', 'all_sensors_need_10_samples'])
    except getopt.GetoptError:
        print('incorrect usage:\n', usage_msg)
        sys.exit(1)

    # parse arguments
    for opt, arg in opts:
        if opt in ('-h', '-?'):
            print(usage_msg)
            sys.exit(0)
        elif opt in ('--data_folder', '-d', '-f', '-df'):
            data_folder = arg
        elif opt in ('--all_sensors_need_10_samples'):
            all_sensors_need_10_samples = True
        
    # validate arguments
    if data_folder is None:
        print('need valid data_folder')
        sys.exit(1)

    # create new datafolder
    balanced_data_folder = data_folder[:-1 if data_folder[-1] == '/' else len(data_folder)] + '_balanced'
    if not os.path.exists(balanced_data_folder):
        os.mkdir(balanced_data_folder)

    copy_files(data_folder, balanced_data_folder, all_sensors_need_10_samples) 

    return

if __name__ == "__main__":
    main()

