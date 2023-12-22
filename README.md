# FingerprintMatching
Code to accompany our *Science Advances* paper *Unveiling the Intra-Person Fingerprint Similarity via Deep Contrastive Learning* by: [Gabe Guo](https://gabeguo.github.io/), [Aniv Ray](https://github.com/AnivRay), [Miles Izydorczak](https://github.com/milesizydorczak12), [Judah Goldfeder](https://judahgoldfeder.com/), [Hod Lipson](https://www.hodlipson.com/), [Wenyao Xu](https://cse.buffalo.edu/~wenyaoxu/). Use of this code by outside parties is only allowed for research purposes.

[![DOI](https://zenodo.org/badge/552203854.svg)](https://zenodo.org/badge/latestdoi/552203854)

# Environment
~~Python 3.6.9 with requirements.txt~~
~~OR, if your hardware does not support it:~~
Python 3.10.0 with requirements_python_3_10_0.txt

# Getting Dataset

Get [SD302](https://www.nist.gov/itl/iad/image-group/nist-special-database-302), [SD301](https://www.nist.gov/itl/iad/image-group/nist-special-database-301), [SD300](https://www.nist.gov/itl/iad/image-group/nist-special-database-301), [RidgeBase](https://www.buffalo.edu/cubs/research/datasets/ridgebase-benchmark-dataset.html).

The person-wise train, test, and val splits are available in [data_split_info](data_split_info).

## NIST SD datasets

To split by person identity after you unzip the dataset (creates copies):

    cd directory_organization
    mkdir dummy
    python gather_nist_data.py --train_src [unzipped dataset folderpath] --test_src dummy --dest [new folderpath for data split by person]

This will give you data (split by PID, in a NEW COPY directory) in whatever you put for ```--dest```.

To split into train, val, and test (IN-PLACE):

    python split_people.py --data_folder [whatever you put for --dest in last command] --train_percent [int 0 -> 100, typically 80] --val_percent [int 0 -> 100]

To get finger-balanced dataset (COPIED from source):

    python get_all_10_finger_samples.py --data_folder [whatever you've been putting for --dest]

This will give you COPIED data in a folder named as: whatever you put for ```--data_folder```, with ```_balanced``` appended to the foldername.

### SD300

Use only SD300a (other parts are higher resolution, which introduces duplicates).

For ```split_people.py```, make sure to include ```--max_fgrp 12``` at the end, so we can include thumbs (they have a different code, due to the collection protocol).

For ```get_all_10_finger_samples.py```, make sure to include option ```--all_sensors_need_10_samples```, so that data is perfectly balanced (otherwise, since there are only two modalities, triplet generation will not converge).

### SD301

Use only SD301a (other datasets are latent images or other stuff). Make sure to move Person 00002239 into train, since that overlaps with SD302.

### SD302

Use only SD302a-d (other datasets are latent images or other stuff).

### RidgeBase

Use only ```Task1/Train``` (from the original directory structure). Before running the other steps, you need to run ```convert_ub_dataset_to_nist_format.py``` (and make sure to change filepaths in the script accordingly).

# Feature Extraction

## Enhanced Images
    cd level2FeatureExtraction
    python extract_minutiae.py --src [source image folder path]

The output will be in a folder in the same directory level as the orignal image folder, named:

    [source image root directory]/img_l2_feature_extractions_[source image foldername]/enhance

Do NOT use the minutiae. (That is legacy code.)

## Frequency and Orientation Maps
Make sure you already extracted enhanced images. 

    cd level1FeatureExtraction
    python main.py --src [orig root directory]/img_l2_feature_extractions_[source image foldername]/enhance

The output will be in:

    [orig root directory]/img_l2_feature_extractions_[source image foldername]/img_l1_feature_extractions/freq
    [orig root directory]/img_l2_feature_extractions_[source image foldername]/img_l1_feature_extractions/orient

## Minutiae

Use [MindTCT](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=51097).

    cd level2FeatureExtraction
    python myLevelTwoExtraction.py --dataset_name [name of dataset] --dataset_root [root of fingerprint image datasets] --mindtct_root [root of mindtct output datasets]

Note: Each root should have a subdirectory with name dataset_name that contains the images/mindtct output.

# Replicating Experiments

**Filepaths will need to be changed in the .sh scripts!**

Do not slash / at the end for folder inputs to bash scripts

## Pretraining
    cd synthetic_data_transfer_learning
    CUDA_VISIBLE_DEVICES=x python3 runner.py \
        --model_path "[desired output folder]/model_weights/embedding_net_weights_printsgan.pth" \
        --data_path "/data/therealgabeguo/printsgan" \
        --output_folder "[desired output folder]/pretrain_results"

## Base Model

### With Pretraining

    cd dl_models
    bash run_base_model_pretrained.sh [desired output_folder (must be same as PrintsGAN weights)] [cuda_num]

### Without Pretraining
    cd dl_models
    bash run_base_model_no_pretrain.sh [desired output folder (no slash at end)] [cuda_num]

## Finger-by-Finger Correlation
    cd dl_models
    bash run_finger_by_finger_balanced.sh [desired output folder] [cuda num]

## Feature Correlations
    cd dl_models
    bash run_feature_correlation.sh [desired output folder] [cuda num]

## Demographics (Generalizability)

### Race

    cd dl_models
    bash run_race.sh [desired_output_folder] [cuda_num]

### Gender (Generalizability)

    cd dl_models
    bash run_gender.sh [desired_output_folder] [cuda_num]

## Performance

### Multi-Finger Testing

Must train (pretrained) base model first!

    cd dl_models
    bash test_multi_fingers.sh [desired_output_folder] [cuda_num]

### Lead Efficiency Analysis

    cd dl_models
    bash distribution_shift_runner_all.sh [cuda_num]

Note: change the line 18 and 19 as follows:

    results_filename="geometric_analysis_results_1.json"
    python distribution_shift_tester_batched.py -d /data/therealgabeguo/fingerprint_data/sd302_split_balanced -w /data/therealgabeguo/embedding_net_weights.pth -p $prior -c $cuda -n $finger -o /data/verifiedanivray/results -s $scale_factor -a 0.95 -dfs -dws -dss -sss -l $results_filename

Change results_filename to your desired output filename. Change -d to your dataset directory, change -w to your embedding weights path, and change -o to your desired output directory    

## Interpretability

### Saliency Maps

    cd Interpretability
    CUDA_VISIBLE_DEVICES=x bash run_gradcam.sh

### Convolutional Filter Visualization

    cd Interpretability
    python cnn_layer_visualization.py --model_path [path to embedder weights] --output_folder [folder where you want output] --conv_layer [convolutional layer id]

The convolutional layer id is given from the ResNet1-18 architecture as follows: x.y.convz where x = block #, y = sub-block (1 or 2), and z = first or second conv layer (1 or 2). For example "7.1.conv2" would represent the 7th block, first sub-block, and second conv2 layer within the sub-block.

## Supplements

### Two-Finger Ablation

    cd dl_models
    bash run_two_finger.sh [desired_output_folder] [cuda_num]
