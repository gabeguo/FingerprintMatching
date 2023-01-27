import os

DATA_FOLDER = '/data/therealgabeguo/most_recent_experiment_reports/jan_20_2023_multiFinger_diffFinger'
line_by_dataset = dict()

for filename in os.listdir(DATA_FOLDER):
    if '.txt' not in filename:
        continue
    
    # get info about the testing dataset
    n_fingers = int(filename[:-4].split('_')[-1])
    the_dataset = filename.split('_')[4]

    # read experiment-specific info
    with open(os.path.join(DATA_FOLDER, filename), 'r') as fin:
        lines = fin.readlines()

        test_folder = os.path.join(lines[0].split()[-1].strip(), 'test')

        n_files = sum([len([the_file for the_file in files \
            if '_1000_' not in the_file and '_2000_' not in the_file]) \
            for r, d, files in os.walk(test_folder)])
        n_classes = len(os.listdir(test_folder))

        roc_auc = float(lines[9].split()[-1])

    # create key
    dataset_key = "{} ({} samples from {} people)".format(the_dataset.upper(), n_files, n_classes)

    # create data storage
    if dataset_key not in line_by_dataset:
        line_by_dataset[dataset_key] = list()
    line_by_dataset[dataset_key].append((n_fingers, roc_auc))

import matplotlib.pyplot as plt

line_markers = ['o', '*']
for key in line_by_dataset:
    curr_data = line_by_dataset[key]
    curr_x = [item[0] for item in curr_data]
    curr_y = [item[1] for item in curr_data]
    plt.plot(curr_x, curr_y, line_markers.pop(0) + '-', label=key)

plt.legend()
plt.xlim(0.75, 5.25)
plt.xticks([i for i in range(1, 6)], ['1-to-1', '2-to-2', '3-to-3', '4-to-4', '5-to-5'])
plt.xlabel('Number of Fingers')
plt.ylim(0.75, 0.95)
plt.ylabel('ROC AUC')
plt.title('N-to-N Disjoint Finger Matching Results')
plt.grid()

plt.savefig(os.path.join('output', 'multi_finger_results.pdf'))
plt.savefig(os.path.join('output', 'multi_finger_results.png'))

plt.show()