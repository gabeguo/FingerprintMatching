"""

This script copies fingerprints scanned by certain sensors
    into a duplicate of the data directory, except it only contains those samples.

It assumes that the original data directory was created via
    split_people.py

Start State:
    data_folder
        /train
            /00002650
                00002650_S_500_slap_03.png
                00002650_H_500_slap_04.png
        /val
            /00002651
                00002651_S_500_slap_07.png
                00002651_A_500_slap_02.png
        /test
            /00002646
                00002646_S_500_slap_05.png
                00002646_K_500_slap_04.png

End State (assume we only want sensor S):
    data_folder_subset
        /train
            /00002650
                00002650_S_500_slap_03.png
        /val
            /00002651
                00002651_S_500_slap_07.png
        /test
            /00002646
                00002646_S_500_slap_05.png

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

"""
Copies the files from data_folder into subset_data_folder that are from desired_sensors
"""
def copy_files(data_folder, subset_data_folder, desired_sensors):
    create_train_val_test_subfolders(subset_data_folder)

    # https://stackoverflow.com/questions/1192978/python-get-relative-path-of-all-files-and-subfolders-in-a-directory
    for dirpath, dirnames, filenames in os.walk(data_folder):
        for filename in filenames:
            if any(ending in filename for ending in ['.jpg', '.png', '.jpeg', '.pneg']) \
                    and get_sensor(filename) in desired_sensors:
                rel_dir = os.path.relpath(dirpath, data_folder)
                rel_file = os.path.join(rel_dir, filename)
                
                orig_path = os.path.join(dirpath, filename)
                new_path = os.path.join(subset_data_folder, rel_file)
                
                # print('\t', orig_path, '\n\t', new_path, '\n')
                shutil.copy(orig_path, new_path)
    return

def main():
    # Variables we need
    data_folder = None
    desired_sensors = None

    argv = sys.argv[1:] 

    # help message
    usage_msg = "Usage: copy_people.py --data_folder <directory address> --desired_sensors <sensors we want to consider>"
    
    try:
        opts, args = getopt.getopt(argv, "h", ['help', 'data_folder=', 'desired_sensors='])
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
        elif opt in ('--desired_sensors', '-s', '-ds'):
            desired_sensors = arg
        
    # validate arguments
    if data_folder is None:
        print('need valid data_folder')
        sys.exit(1)
    if desired_sensors is None:
        print('need desired sensors')
        sys.exit(1)

    # convert desired_sensors to list form
    desired_sensors = desired_sensors.split()
    print(desired_sensors)

    # create new datafolder
    subset_data_folder = data_folder[:-1 if data_folder[-1] == '/' else len(data_folder)] + '_subset'
    if not os.path.exists(subset_data_folder):
        os.mkdir(subset_data_folder)

    copy_files(data_folder, subset_data_folder, desired_sensors) 

    return

if __name__ == "__main__":
    main()

