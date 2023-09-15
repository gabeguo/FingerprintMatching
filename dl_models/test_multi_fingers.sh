# Usage: test_multi_fingers.sh output_dir cuda_num
# Purpose: To see how our identification performance improves when we add more fingers

BASED_WEIGHTS=$1/model_weights/full_based_model_PRETRAINED.pth # saved to a different path than un-pretrained

SD302_BALANCED='/data/therealgabeguo/fingerprint_data/sd302_split_balanced'
SD301_BALANCED='/data/therealgabeguo/fingerprint_data/sd301_split_balanced'
SD300_BALANCED='/data/therealgabeguo/fingerprint_data/sd300a_split_balanced'

##############################
### TEST MULTIPLE FINGERS
###############
MULTIPLE_FINGER_FOLDER="$1/paper_results/multi_finger"
datasets=($SD302_BALANCED $SD301_BALANCED $SD300_BALANCED)
names=("sd302_balanced" "sd301_balanced" "sd300_balanced")
num_datasets=${#datasets[@]}
for j in 1 2 3
do
    for i in 1 2 3 4 5
    do
        # Loop over both arrays simultaneously
        for ((idx=0; idx<$num_datasets; idx++))
        do
            curr_dataset=${datasets[idx]}
            curr_name=${names[idx]}
            python3 parameterized_multiple_finger_tester.py \
                --dataset $curr_dataset \
                --weights $BASED_MODEL \
                --cuda "cuda:$2" \
                --num_fingers $i \
                --output_root $MULTIPLE_FINGER_FOLDER/$curr_name/$i \
                --scale_factor 1 \
                --diff_fingers_across_sets \
                --diff_fingers_within_set \
                --diff_sensors_across_sets \
                --same_sensor_within_set
        done
    done
done