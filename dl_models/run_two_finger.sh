# Usage: bash run_two_finger.sh output_dir cuda_num
# Purpose: to see how our model performs, when training with only two fingers

SD302='/data/therealgabeguo/fingerprint_data/sd302_split'
SD300='/data/therealgabeguo/fingerprint_data/sd300a_split'
RIDGEBASE='/data/therealgabeguo/fingerprint_data/RidgeBase_Split'

DUAL_MODEL=$1/model_weights/dual_02_03.pth

######
# Two-finger experiment
######
CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
    --datasets "${SD302} ${SD300} ${RIDGEBASE}" \
    --val-datasets "${SD302}" \
    --possible-fgrps '02 03' \
    --posttrained-model-path $DUAL_MODEL \
    --temp_model_dir 'temp_weights_ablation' \
    --results_dir "$1/results" \
    --diff-sensors-across-sets-train \
    --diff-sensors-across-sets-val \
    --scale-factor 4 \
    --log-interval 20

python3 parameterized_multiple_finger_tester.py \
    --dataset $SD302 \
    --weights $DUAL_MODEL \
    --cuda "cuda:$2" \
    --num_fingers 1 \
    --output_root $1/paper_results/generalizing_sub_model/sd302 \
    --scale_factor 4 \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set \