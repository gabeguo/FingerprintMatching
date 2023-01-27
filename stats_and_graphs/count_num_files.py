import os
import argparse

# counts number of fingerprint images
def main(data_folder, count_high_res=False):
    n_files = sum([len([the_file for the_file in files \
        if ('.png' in the_file or '.jpg' in the_file or '.bmp' in the_file) \
        and (count_high_res or ('_1000_' not in the_file and '_2000_' not in the_file)) \
        ]) \
        for r, d, files in os.walk(data_folder)])
    n_classes = len(os.listdir(data_folder))

    print('{} has {} classes, {} files'.format(data_folder, n_classes, n_files))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser('count_num_files.py')
    parser.add_argument('--dataset', '-d', help='Path to folders containing images', type=str)
    parser.add_argument('--high_res', '-r', help='Count duplicate high res (1000, 2000 px) images', action='store_true')
    args = parser.parse_args()
    dataset = args.dataset
    high_res = args.high_res

    main(dataset, high_res)