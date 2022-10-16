"""

This script does the train-test-val split.
It assumes that all the items were already in the train folder, and arranged in subfolders by class name.
Ex:
    data_folder
        /train
            /class1
                sample.jpg
                another_sample.jpg
            /class2

"""

if __name__ == "__main__":
    # Usage: 
