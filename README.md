# FingerprintMatching
Project to match different fingerprints from the same person.

# Environment
~~Python 3.6.9 with requirements.txt~~
~~OR, if your hardware does not support it:~~
Python 3.10.0 with requirements_python_3_10_0.txt

# Getting Dataset

tbd

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

# Replicating Experiments

filepaths will need to be changed in the .sh scripts

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

## Supplements

### Two-Finger Ablation

    cd dl_models
    bash run_two_finger.sh [desired_output_folder] [cuda_num]
