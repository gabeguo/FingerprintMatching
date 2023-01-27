import os
import argparse

# counts number of fingerprint images, excluding duplicates
def main(data_folder):
    n_files = sum([len([the_file for the_file in files \
        if ('.png' in the_file or '.jpg' in the_file) \
        and '_1000_' not in the_file and '_2000_' not in the_file]) \
        for r, d, files in os.walk(data_folder)])
    n_classes = len(os.listdir(data_folder))

    print('{} has {} classes, {} files'.format(data_folder, n_classes, n_files))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser('count_num_files.py')
    parser.add_argument('--dataset', '-d', help='Path to folders containing images', type=str)
    args = parser.parse_args()
    dataset = args.dataset

    main(dataset)