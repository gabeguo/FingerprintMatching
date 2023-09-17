# Usage: bash train_balanced.sh root_output_directory cuda_num
# Purpose: to investigate finger-by-finger correlation

# Only balanced datasets here!
SD302_BALANCED='/data/therealgabeguo/fingerprint_data/sd302_split_balanced'
SD301_BALANCED='/data/therealgabeguo/fingerprint_data/sd301_split_balanced/'
SD300_BALANCED='/data/therealgabeguo/fingerprint_data/sd300a_split_balanced/'

# Where to output
BALANCED_WEIGHTS=$1/model_weights/balanced_model_sd302_sd300.pth
PROVING_CORRELATION_FOLDER="$1/paper_results/proving_correlation"
FINGER_CORRELATION_FOLDER="$PROVING_CORRELATION_FOLDER/finger-by-finger"

######
# Training balanced model
######
CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
    --datasets "${SD302_BALANCED} ${SD300_BALANCED}" \
    --val-datasets "${SD302_BALANCED} ${SD300_BALANCED}" \
    --posttrained-model-path $BALANCED_WEIGHTS \
    --temp_model_dir 'temp_weights' --results_dir "$1/results" \
    --diff-sensors-across-sets-train --diff-sensors-across-sets-val \
    --scale-factor 1 --log-interval 100
# When training and validating balanced model, we're allowed to have same finger in anchor and pos/neg set
# Note that we only use balanced versions of datasets

######
# Testing balanced model
# for finger-by-finger correlation
######
# Array of datasets
datasets=($SD300_BALANCED $SD301_BALANCED $SD302_BALANCED) # Only balanced datasets!
# Array of output roots for general and finger correlation (note that they're balanced)
folder_names=("sd300_balanced" "sd301_balanced" "sd302_balanced")
# Array of scale factors
scale_factors=(2 4 4) # SD300 has many more images, so need less scaling

for ((i=0; i<${#datasets[@]}; ++i)); do
    # Finger correlation
    python3 parameterized_multiple_finger_tester.py \
        --dataset ${datasets[i]} \
        --weights $BALANCED_WEIGHTS \
        --cuda "cuda:$2" \
        --num_fingers 1 \
        --output_root "${FINGER_CORRELATION_FOLDER}"/${folder_names[i]} \
        --scale_factor ${scale_factors[i]} \
        --diff_fingers_within_set \
        --diff_sensors_across_sets \
        --same_sensor_within_set
    # we allow same finger across sets, since we want to see finger-by-finger
done