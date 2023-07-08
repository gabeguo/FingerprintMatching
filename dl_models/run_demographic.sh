# run_demographic.sh root_directory cuda

DEMOGRAPHICS_ROOT="/data/verifiedanivray/demographics_datasets"
CAUCASIAN_DESCENT="sd302_white_split"
NON_CAUCASIAN="sd302_non-white_split"
MALE_GROUP="sd302_male_split"
FEMALE_GROUP="sd302_female_split"

######
# Training demographic models
######
for category in $CAUCASIAN_DESCENT $NON_CAUCASIAN $MALE_GROUP $FEMALE_GROUP
do
    folder="${DEMOGRAPHICS_ROOT}/${category}"
    CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
        --datasets "${folder}" \
        --val-datasets "${folder}" \
        --posttrained-model-path $1/model_weights/demographic_model_${category}.pth \
        --temp_model_dir 'temp_weights' --results_dir "$1/results" \
        --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
        --scale-factor 1 --log-interval 100
done

##############################
### DEMOGRAPHIC FAIRNESS
###############
FAIRNESS_OUTPUT_FOLDER=$1/paper_results/fairness
### Race
for train_group in $CAUCASIAN_DESCENT $NON_CAUCASIAN
do
    for test_group in $CAUCASIAN_DESCENT $NON_CAUCASIAN
    do
        demographic_model=$1/model_weights/demographic_model_${train_group}.pth
        demographic_folder="${DEMOGRAPHICS_ROOT}/${test_group}"
        python3 parameterized_multiple_finger_tester.py \
            --dataset $demographic_folder \
            --weights $demographic_model \
            --cuda "cuda:$2" \
            --num_fingers 1 \
            --output_root "$FAIRNESS_OUTPUT_FOLDER/sd302/train_${train_group}_test_${test_group}" \
            --scale_factor 2 \
            --diff_fingers_across_sets \
            --diff_fingers_within_set \
            --diff_sensors_across_sets \
            --same_sensor_within_set
    done
done
### Gender
for train_group in $MALE_GROUP $FEMALE_GROUP
do
    for test_group in $MALE_GROUP $FEMALE_GROUP
    do
        demographic_model=$1/model_weights/demographic_model_${train_group}.pth
        demographic_folder="${DEMOGRAPHICS_ROOT}/${test_group}"
        python3 parameterized_multiple_finger_tester.py \
            --dataset $demographic_folder \
            --weights $demographic_model \
            --cuda "cuda:$2" \
            --num_fingers 1 \
            --output_root "$FAIRNESS_OUTPUT_FOLDER/sd302/train_${train_group}_test_${test_group}" \
            --scale_factor 2 \
            --diff_fingers_across_sets \
            --diff_fingers_within_set \
            --diff_sensors_across_sets \
            --same_sensor_within_set
    done
done