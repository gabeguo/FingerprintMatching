# FingerprintMatching
Project to match different fingerprints from the same person.

# Environment
Python 3.6.9 with requirements.txt

OR, if your hardware does not support it:

Python 3.10.0 with requirements_python_3_10_0.txt

# Getting Dataset

tbd

# Replicating Experiments

filepaths will need to be changed in the .sh scripts

Do not slash \\ at the end for folder inputs to bash scripts

## Finger-by-Finger Correlation
    cd dl_models
    bash run_finger_by_finger_balanced.sh \[desired output folder (no slash at end)\] \[cuda num\]

## Demographics (Generalizability)

### Race

    cd dl_models
    bash run_race.sh \[desired_output_folder\] \[cuda_num\]

### Gender (Generalizability)

    cd dl_models
    bash run_gender.sh \[desired_output_folder\] \[cuda_num\]