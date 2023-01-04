'''
gather_nist_data.py

Given source and destination directories, filters the relevant image files from
the source and stores them in the destination organized in folders by person.

If no destination is given creates a folder named 'train' in the current one to
use as the destination.

USAGE NOTICE: Make a blank dummy folder to input as test.
Specify all the desired folders to be training.
Because for this project, we're doing image matching (Siamese architecture). 
A separate script will split training people from testing people. 
This script is a holdover from the proof-of-concept project that proved correlation - 
in that one, we had same people in train and test. 
In this one, different people will be in train and test.
'''

import sys, getopt, os, shutil
from fileProcessingUtil import *

def renameFile(filepath):
    old_filepath = filepath
    print('old:', old_filepath)
    tokens = filepath.split('/')
    filename = tokens[-1]
    fgrp = get_fgrp(filename)
    assert int(fgrp) <= 12
    assert fgrp == '11' or fgrp == '12'
    if int(fgrp) > 10:
        old_fgrp = fgrp
        # '11' & '01' are right thumbs; '12' and '06' are left thumbs
        fgrp = '01' if fgrp == '11' else '06' #'{:02d}'.format(int(fgrp) % 10)
        #print(fgrp)
        filename = filename[:-6] + 'MOD' + old_fgrp + '_' + fgrp + filename[-4:]
        #print(filename)
        tokens[-1] = filename
        new_filepath = '/'.join(tokens)
        print('new:', new_filepath)
        #return filepath
        os.rename(old_filepath, new_filepath)

def copyFiles(TRAIN_NIST_DATA_DIRS:list, TEST_NIST_DATA_DIRS:list, DEEP_LEARNING_DIR:str, max_fgrp=10):
    num_samples = {x:0 for x in ['train', 'val']}
    num_copied = {x:0 for x in ['train', 'val']}
    num_missed = {x:0 for x in ['train', 'val']}
    num_nonFgrp = {x:0 for x in ['train', 'val']}

    fgrp2count = {'train':dict(), 'val':dict()}
    id2count = {'train':dict(), 'val':dict()}
    id2validCount = {'train':dict(), 'val':dict()}

    for NIST_DATA_DIRS, group in zip([TRAIN_NIST_DATA_DIRS, TEST_NIST_DATA_DIRS], ['train', 'val']):
        for NIST_DATA_DIR in NIST_DATA_DIRS:
            for (root, dirs, files) in os.walk(NIST_DATA_DIR, topdown=True):
                for filename in files:
                    if '.png' in filename or '.jpg' in filename or '.bmp' in filename:
                        pid = get_id(filename)
                        fgrp = get_fgrp(filename)
                        num_samples[group] += 1

                        # get totals for each finger print
                        if fgrp not in fgrp2count[group]:
                            fgrp2count[group][fgrp] = 0
                        fgrp2count[group][fgrp] += 1

                        # get totals for each person
                        if pid not in id2count[group]:
                            id2count[group][pid] = 0
                        id2count[group][pid] += 1

                        src = os.path.join(root, filename)

                        # make a folder for each person
                        dest_folder = os.path.join(DEEP_LEARNING_DIR, group, pid)
                        os.makedirs(dest_folder, exist_ok = True)

                        dest = os.path.join(dest_folder, filename)

                        # Get all FGRPs <= max_fgrp (default 10)
                        if int(fgrp) <= max_fgrp:
                            if os.path.exists(src[:-len(filename)]) and os.path.exists(dest[:-len(filename)]):
                                shutil.copyfile(src, dest)
                                num_copied[group] += 1
                                if int(fgrp) > 10:
                                    print('renaming:', src)
                                    renameFile(dest)
                                if pid not in id2validCount[group]:
                                    id2validCount[group][pid] = 0
                                id2validCount[group][pid] += 1
                            else:
                                print('INVALID PATHS')
                        else:
                            num_nonFgrp[group] += 1
                    if '.jpg' in filename or '.pneg' in filename or '.jpeg' in filename \
                    or '.JPG' in filename or '.PNEG' in filename or '.JPEG' in filename \
                    or '.PNG' in filename:
                        print('MISSED FILE')
                        num_missed[group] += 1
    return num_samples, num_copied, num_missed, num_nonFgrp, fgrp2count, id2count, id2validCount

def main(argv):

    TRAIN_NIST_DATA_DIRS = TEST_NIST_DATA_DIRS = ''
    DEEP_LEARNING_DIR = ''
    MAX_FGRP = 10 # default 10 (don't include higher samples)
    print_summary = False

    usage_msg = "Usage: gather_nist_data.py --train_src <string of training_src_dirs separated by spaces> --test_src <string of testing_src_dirs separated by spaces> \n" + \
        "--dest <dest_dir> --max_fgrp <max_fgrp> [--summary]"

    try:
        opts, args = getopt.getopt(argv,"h?s",["help", "summary", "train_src=", "test_src=", "dest=", "max_fgrp="])
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-h', '-?', '--help'):
            print(usage_msg)
            sys.exit(0)
        elif opt in ('--train_src', '--train', '-r'):
            TRAIN_NIST_DATA_DIRS = arg.split()
        elif opt in ('--test_src', '--test', '-e'):
            TEST_NIST_DATA_DIRS = arg.split()
        elif opt in ('--dest', '--dest_dir', '-d'):
            DEEP_LEARNING_DIR = arg
        elif opt in ('--max_fgrp', '-mf'):
            MAX_FGRP = int(arg)
        elif opt in ('--summary', '-s'):
            print_summary = True

    # Garuntee that user has provided src
    if not TRAIN_NIST_DATA_DIRS or not TEST_NIST_DATA_DIRS or not DEEP_LEARNING_DIR:
        print(usage_msg)
        sys.exit(1)

    # Garuntee that src and dest are real
    for dirs in (TRAIN_NIST_DATA_DIRS, TEST_NIST_DATA_DIRS):
        for dir in dirs:
            if not os.path.exists(dir):
                print(dir, "does not exist")
                sys.exit(1)
    if not os.path.exists(DEEP_LEARNING_DIR):
        os.makedirs(DEEP_LEARNING_DIR, exist_ok=True)
        print(DEEP_LEARNING_DIR, "does not exist, so we create it")
        #sys.exit(1)

    num_samples, num_copied, num_missed, num_nonFgrp, fgrp2count, id2count, id2validCount = \
    copyFiles(TRAIN_NIST_DATA_DIRS, TEST_NIST_DATA_DIRS, DEEP_LEARNING_DIR, max_fgrp=MAX_FGRP)

    if print_summary:
        print('Both groups contain same people:', id2count['train'].keys() == id2count['val'].keys())
        for group in ['train', 'val']:
            print("\n\nSummary for Group:", group)
            print('\tnumber of samples:', num_samples[group])
            print('\tnumber of files copied:', num_copied[group])
            print('\tnumber of missed files:', num_missed[group])
            print('\tnumber of non-fingerprints:', num_nonFgrp[group])

            # CHECK
            num_successful = 0
            for (root, dirs, files) in os.walk(os.path.join(DEEP_LEARNING_DIR, group)):
                for filename in files:
                    if '.png' in filename:
                        num_successful += 1

            print('\tnumber of successful copies:', num_successful)

            print('\tnumber of people:', len(id2count[group]))

            print('\tsamples by pid:')
            for id in sorted(id2count[group]):
                print('\t\t' + id + ':', id2count[group][id])
            print('\tsamples by fgrp:')
            for fgrp in sorted(fgrp2count[group]):
               print('\t\t' + fgrp + ':', fgrp2count[group][fgrp])
            print('\tvalid samples by pid:')
            for id in sorted(id2validCount[group]):
                print('\t\t' + id + ':', id2validCount[group][id])

    with open(DEEP_LEARNING_DIR + '/input.txt', 'w') as fout:
        fout.write(str(argv))

if __name__ == "__main__":
   main(sys.argv[1:])
