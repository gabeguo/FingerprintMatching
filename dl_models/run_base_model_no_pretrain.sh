# Usage: bash run_base_model_no_pretrain.sh output_folder cuda_num
# Purpose: to train our cross-finger recognition model, without pretraining

SD302='/home/gabeguo/fingerprint_data/sd302_split'
SD301='/home/gabeguo/fingerprint_data/sd301_split'
SD300='/home/gabeguo/fingerprint_data/sd300a_split'
RIDGEBASE='/home/gabeguo/fingerprint_data/RidgeBase_Split'
BASED_WEIGHTS=$1/model_weights/full_based_model.pth
PROVING_CORRELATION_FOLDER="$1/paper_results/proving_correlation"
GENERAL_CORRELATION_FOLDER="$PROVING_CORRELATION_FOLDER/general"

######
# Training base model
######
CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
    --datasets "${SD302} ${SD300} ${RIDGEBASE}" \
    --val-datasets "${SD302} ${SD300} ${RIDGEBASE}" \
    --posttrained-model-path $BASED_WEIGHTS \
    --temp_model_dir 'temp_weights' --results_dir "$1/results" \
    --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
    --scale-factor 1 --log-interval 100

######
# Testing base model
######
# Array of datasets
datasets=($SD300 $SD301 $SD302)
# Array of output roots for general correlation
folder_names=("sd300_full" "sd301_full" "sd302_full")
# Array of scale factors
scale_factors=(2 4 4) # SD300 has many more images, so need less scaling

for ((i=0; i<${#datasets[@]}; ++i)); do
    # Finger correlation
    python3 parameterized_multiple_finger_tester.py \
        --dataset ${datasets[i]} \
        --weights $BASED_WEIGHTS \
        --cuda "cuda:$2" \
        --num_fingers 1 \
        --output_root "${GENERAL_CORRELATION_FOLDER}"/${folder_names[i]} \
        --scale_factor ${scale_factors[i]} \
        --diff_fingers_across_sets \
        --diff_fingers_within_set \
        --diff_sensors_across_sets \
        --same_sensor_within_set
    # Only cross-finger testing!
done