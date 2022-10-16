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
            /class 1
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

if __name__ == "__main__":
    usage_msg = "Usage: split_people.py --train_percent <(0, 1)> --val_percent <(0, 1)>" + \
            "\nNote: train_percent + val_percent < 1, test_percent = 1 - (train_percent + val_percent)"
