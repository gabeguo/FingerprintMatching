# Thanks ChatGPT!

from collections import defaultdict
from fileProcessingUtil import get_id, get_fgrp
import argparse

import os

def count_matching_fingers(filename):
    matches = dict()
    num_matches = defaultdict(int)

    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        path1, path2, score = line.split()
        filename1 = os.path.basename(path1)
        filename2 = os.path.basename(path2)

        assert filename1[-4:] == '.xyt'
        assert filename2[-4:] == '.xyt'
        filename1 = filename1[:-4]
        filename2 = filename2[:-4]
        assert filename1[-4:] == '.png'
        assert filename2[-4:] == '.png'

        person1, finger1 = get_id(filename1), get_fgrp(filename1)
        person2, finger2 = get_id(filename2), get_fgrp(filename2)

        if finger1 == finger2 and person1 != person2:
            pair = tuple(sorted((person1, person2)))  # Create a pair
            if pair not in matches:
                matches[pair] = set()
            matches[pair].add(finger1)
            num_matches[pair] += 1

    return matches, num_matches

def main():
    parser = argparse.ArgumentParser(description='Count matching fingers between pairs.')
    parser.add_argument('--filename', type=str, help='Path to the input text file.')
    parser.add_argument('--finger_threshold', type=int, default=1, help='Min number of fingers to be considered same person')

    args = parser.parse_args()

    print('Pairs of people with over four matches:')
    matches, num_matches = count_matching_fingers(args.filename)
    for pair, curr_matching_fingers in matches.items():
        if len(curr_matching_fingers) >= args.finger_threshold:
            print(f'Pair {pair}: {curr_matching_fingers} matching ({len(curr_matching_fingers)} count)')
            print(f'\t{num_matches[pair]} matching samples')

if __name__ == "__main__":
    main()