import os
import random
import pandas as pd
import argparse

def getStats(demographicsDf):
    raceCol = set(demographicsDf["race"].tolist())
    print(raceCol)
    for race in raceCol:
        print(race, " ", demographicsDf[demographicsDf.race == race].shape[0])
    print("\n")
    genderCol = set(demographicsDf["gender"].tolist())
    print(genderCol)
    for gender in genderCol:
        print(gender, " ", demographicsDf[demographicsDf.gender == gender].shape[0])

def main(csvPath, cat, val, sampleSize, inputRoot, outputRoot, trainProp, valProp):
    demographicsDf = pd.read_csv(csvPath)
    
    if val == "non-white":
        personIds = demographicsDf[demographicsDf[cat] != "white"][demographicsDf[cat] != "no answer"]["id"].tolist()
    else:
        personIds = demographicsDf[demographicsDf[cat] == val]["id"].tolist()
    assert sampleSize <= len(personIds)
    selectedPersonIds = random.sample(personIds, sampleSize)
    print(selectedPersonIds)
    outputDatasetPath = os.path.join(outputRoot, "sd302_{}_split".format(val))
    if not os.path.exists(outputDatasetPath):
        os.mkdir(outputDatasetPath)
    outputPath = os.path.join(outputDatasetPath, "train")
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    for personId in selectedPersonIds:
        if os.path.exists(os.path.join(inputRoot, "train", str(personId).zfill(8))):
            inputPersonPath = os.path.join(inputRoot, "train", str(personId).zfill(8))
        elif os.path.exists(os.path.join(inputRoot, "val", str(personId).zfill(8))):
            inputPersonPath = os.path.join(inputRoot, "val", str(personId).zfill(8))
        else:
            inputPersonPath = os.path.join(inputRoot, "test", str(personId).zfill(8))
        assert os.path.exists(inputPersonPath)
        os.system("cp -r {} {}".format(inputPersonPath, outputPath))
    print(int(trainProp*100), int(valProp*100))
    os.system("python split_people.py --data_folder {} --train_percent {} --val_percent {}".format(outputDatasetPath, int(trainProp*100), int(valProp*100)))

# python demographic_dataset_builder.py -d=/data/therealgabeguo/fingerprint_data/sd302_split -dc=race -dv=white -s=62 -c=/data/therealgabeguo/fingerprint_data/sd302_split/participants.csv -o/data/verifiedanivray/demographics_datasets -t=0.8 -v=0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="demographics.py")
    parser.add_argument('--dataset', '-d', help='Path to folders containing images', type=str)
    parser.add_argument('--demographic_cat', '-dc', help='Which demographic category (e.g. race, gender)', type=str)
    parser.add_argument('--demographic_val', '-dv', help='Which demographic value for the given class (e.g. white, female)', type=str)
    parser.add_argument('--sample_size', '-s', help='How many people to sample for the given demographic value', type=int)
    parser.add_argument('--demographics_csv', '-c', help='Path to demographics csv file', type=str)
    parser.add_argument('--output_root', '-o', nargs='?', help='Root directory where new dataset directory will be created', type=str)
    parser.add_argument('--train_prop', '-t', help='Proportion of samples in training data', type=float)
    parser.add_argument('--val_prop', '-v', help='Proportion of samples in validation data', type=float)
    args = parser.parse_args()
    
    assert args.train_prop + args.val_prop < 1.0

    main(args.demographics_csv, args.demographic_cat, args.demographic_val, args.sample_size, args.dataset, args.output_root, args.train_prop, args.val_prop)

    