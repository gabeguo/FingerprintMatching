"""

This script does the train-test-val split.
It assumes that all the items were already in the train folder, 
    and arranged in subfolders by class name.
    If all items weren't in train folder already, it moves them all there.
Keeps some classes in training folder, 
    moves some to val folder, 
    moves some to test folder.
We have disjoint classes in train, val, test; 
    as we are doing siamese neural networks to find fingerprint similarity.

Ex:
Start State:
    data_folder
        /train
            /class1
                class1sample1.jpg
                class1sample2.jpg
            /class2
                class2sample1.jpg
                class2sample2.jpg
            /class3
                class3sample1.jpg
                class3sample2.jpg

End State:
    data_folder
        /train
            /class1
                class1sample1.jpg
                class1sample2.jpg
        /val
            /class2
                class2sample1.jpg
                class2sample2.jpg
        /test
            /class3
                class3sample1.jpg
                class3sample2.jpg

"""

import sys, getopt, os, shutil

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

def create_train_val_test_subfolders(data_folder):
    for subfolder in [TRAIN, VAL, TEST]:
        subdirectory_abs_path = os.path.join(data_folder, subfolder)
        if not os.path.exists(subdirectory_abs_path):
            os.mkdir(subdirectory_abs_path)
    return

def move_all_items_to_train(data_folder):
    for subfolder in [VAL, TEST]:
        subdirectory_abs_path = os.path.join(data_folder, subfolder)
        for pid in os.listdir(subdirectory_abs_path):
            # folder that contains all samples for a person
            pid_full_old_path = os.path.join(subdirectory_abs_path, pid)
            pid_full_new_path = os.path.join(data_folder, 'train', pid)
            shutil.move(pid_full_old_path, pid_full_new_path)
    return
    
def split_by_ratios(data_folder, train_percent, val_percent, test_percent):
    pids = [x for x in os.listdir(os.path.join(data_folder, TRAIN))]
    #random.shuffle(pids)
    train_end_index = int(train_percent * len(pids) / 100)
    val_end_index = train_end_index + int(val_percent * len(pids) / 100)
    train_pids = pids[:train_end_index]
    val_pids = pids[train_end_index:val_end_index]
    test_pids = pids[val_end_index:]
        
    train_dir_abs_path = os.path.join(data_folder, TRAIN)
    for subfolder in [VAL, TEST]:
        subdirectory_abs_path = os.path.join(data_folder, subfolder)
        curr_pids = val_pids if subfolder == VAL else test_pids

        for pid in curr_pids:
            pid_full_old_path = os.path.join(train_dir_abs_path, pid)
            pid_full_new_path = os.path.join(subdriectory_abs_path, pid)
            shutil.move(pid_full_old_path, pid_full_new_path)
    return
        

"""
Given data_folder, splits items by train_percent-val_percent-test_percent
"""
def split_files(data_folder, train_percent, val_percent, test_percent):
    create_train_val_test_subfolders(data_folder)
    move_all_items_to_train(data_folder)
    split_by_ratios(data_folder, train_percent, val_percent, test_percent)
    return

if __name__ == "__main__":
    # Variables we need
    data_folder = None
    train_percent = None
    val_percent = None
    test_percent = None
    
    # help message
    usage_msg = "Usage: split_people.py --data_folder <directory address> --train_percent <(0, 100)> --val_percent <(0, 100)>" + \
            "\nNote: 0 < train_percent + val_percent < 100, test_percent = 100 - (train_percent + val_percent)"
    try:
        opts, args = getopt.getopt(argv, "h", ['help', 'data_folder=', 'train_percent=', 'val_percent='])
    except getopt.GetoptError:
        print('incorrect usage:\n', usage_msg)
        sys.exit(1)

    # parse arguments
    for opt, arg in opts:
        if opt in ('-h', '-?'):
            print(usage_msg)
            sys.exit(0)
        elif opt in ('--data_folder', '-d', '-df'):
            data_folder = arg
        elif opt in ('--train_percent', '-tp', '-tr', '-t'):
            try:
                train_percent = int(arg)
            except ValueError:
                print('please give valid value for train_percent')
                sys.exit(1)
        elif opt in ('--val_percent', '-vp', '-vr', '-v'):
            try:
                val_percent = int(arg)
            except ValueError:
                print('please give valid value for val_percent')
                sys.exit(1)
        
    # validate arguments
    if data_folder is None:
        print('need valid data_folder')
        sys.exit(1)
    if train_percent is None or train_percent <= 0 or train_percent >= 100:
        print('need valid train_percent')
    if val_percent is None or val_percent <= 0 or val_percent >= 100:
        print('need valid val_percent')
    if train_percent + val_percent >= 100 or train_percent + val_percent <= 0:
        print('train_percent + val_percent must be in (0, 100)')

    # calculate test_percent
    test_percent = 100 - (train_percent + val_percent)

    split_files(data_folder, train_percent, val_percent, test_percent)

    return
