# sh run_tests.sh root_folder cuda

# TODO: DOUBLE CHECK CODE!

SD302='/data/therealgabeguo/fingerprint_data/sd302_split'
SD301='/data/therealgabeguo/fingerprint_data/sd301_split'
SD300='/data/therealgabeguo/fingerprint_data/sd300a_split'
RIDGEBASE='/data/therealgabeguo/fingerprint_data/RidgeBase_Split'
SD302_BALANCED='/data/therealgabeguo/fingerprint_data/sd302_split_balanced'
SD301_BALANCED='/data/therealgabeguo/fingerprint_data/sd301_split_balanced/'
SD300_BALANCED='/data/therealgabeguo/fingerprint_data/sd300a_split_balanced/'

DEMOGRAPHICS_ROOT="/data/verifiedanivray/demographics_datasets"
CAUCASIAN_DESCENT="sd302_white_split"
NON_CAUCASIAN="sd302_non-white_split"
MALE_GROUP="sd302_male_split"
FEMALE_GROUP="sd302_female_split"

BASED_MODEL="$1/model_weights/based_model.pth"
BALANCED_MODEL="$1/model_weights/balanced_model_sd302_sd300.pth"
DUAL_MODEL="$1/model_weights/dual_02_03.pth"

FEATURE_EXTRACTIONS_ROOT="/data/therealgabeguo/fingerprint_data/sd302_feature_extractions"
ENHANCED="enhanced"
ORIENTATION="orient"
FREQUENCY="freq"

MINDTCT_MINUTIAE_ROOT="/data/therealgabeguo/fingerprint_data/mindtct_minutiae/sd302"

##############################
### PROVE THE CORRELATION
###############
PROVING_CORRELATION_FOLDER="$1/paper_results/proving_correlation"
GENERAL_CORRELATION_FOLDER="$PROVING_CORRELATION_FOLDER/general"
FINGER_CORRELATION_FOLDER="$PROVING_CORRELATION_FOLDER/finger-by-finger"

# Array of datasets
datasets=("$SD300" "$SD301" "$SD302")
# Array of weights
weights=("$BASED_MODEL" "$BALANCED_MODEL")
# Array of output roots for general and finger correlation
folder_names=("sd300" "sd301" "sd302")

# Array of scale factors
scale_factors=(2 4 4)

# Cuda and number of fingers, assuming these are constants
cuda="cuda:$2"
num_fingers=1

# Iterate through arrays
for ((i=0; i<${#datasets[@]}; ++i)); do
    # General correlation
    python3 parameterized_multiple_finger_tester.py \
        --dataset ${datasets[i]} \
        --weights ${weights[0]} \
        --cuda $cuda \
        --num_fingers $num_fingers \
        --output_root "${GENERAL_CORRELATION_FOLDER}"/${folder_names[i]} \
        --scale_factor ${scale_factors[i]} \
        --diff_fingers_across_sets \
        --diff_fingers_within_set \
        --diff_sensors_across_sets \
        --same_sensor_within_set

    # Finger correlation
    python3 parameterized_multiple_finger_tester.py \
        --dataset ${datasets[i]} \
        --weights ${weights[1]} \
        --cuda $cuda \
        --num_fingers $num_fingers \
        --output_root "${FINGER_CORRELATION_FOLDER}"/${folder_names[i]} \
        --scale_factor ${scale_factors[i]} \
        --diff_fingers_within_set \
        --diff_sensors_across_sets \
        --same_sensor_within_set
done

##############################
### EXAMINE THE SPECIFIC FEATURES THAT ARE CORRELATED
###############
FEATURE_CORRELATION_FOLDER="$1/paper_results/feature_correlation/sd302"
# Specific correlation: SD302 (un-pretrained)
python3 parameterized_multiple_finger_tester.py \
    --dataset $SD302 \
    --weights $1/model_weights/unpretrained_model_sd302.pth \
    --cuda "cuda:$2" \
    --num_fingers 1 \
    --output_root $FEATURE_CORRELATION_FOLDER/baseline_noPretrain \
    --scale_factor 4 \
    --diff_fingers_across_sets \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set
# Specific correlation: SD302 (minutiae)
python3 parameterized_multiple_finger_tester.py \
    --dataset $MINDTCT_MINUTIAE_ROOT \
    --weights $1/model_weights/feature_model_minutiae.pth \
    --cuda "cuda:$2" \
    --num_fingers 1 \
    --output_root $FEATURE_CORRELATION_FOLDER/minutiae \
    --scale_factor 4 \
    --diff_fingers_across_sets \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set
# Specific correlations: SD302 (enhanced, frequency, orientation)
for feature in $ENHANCED $FREQUENCY $ORIENTATION
do
    python3 parameterized_multiple_finger_tester.py \
        --dataset ${FEATURE_EXTRACTIONS_ROOT}/${feature} \
        --weights $1/model_weights/feature_model_${feature}.pth \
        --cuda "cuda:$2" \
        --num_fingers 1 \
        --output_root $FEATURE_CORRELATION_FOLDER/$feature \
        --scale_factor 4 \
        --diff_fingers_across_sets \
        --diff_fingers_within_set \
        --diff_sensors_across_sets \
        --same_sensor_within_set
done

##############################
### TEST MULTIPLE FINGERS
###############
MULTIPLE_FINGER_FOLDER="$1/paper_results/multi_finger"
datasets=($SD302_BALANCED $SD301_BALANCED $SD300_BALANCED)
names=("sd302" "sd301" "sd300")
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

##############################
### DEMOGRAPHIC FAIRNESS
###############
FAIRNESS_OUTPUT_FOLDER=$1/paper_results/fairness

# Array of groups
groups=( "$CAUCASIAN_DESCENT" "$NON_CAUCASIAN" "$MALE_GROUP" "$FEMALE_GROUP" )

# Cuda and number of fingers, assuming these are constants
cuda="cuda:$2"
num_fingers=1

# Iterate over the groups
for ((i=0; i<${#groups[@]}; i+=2)); do
    for train_group in "${groups[i]}" "${groups[i+1]}"; do
        for test_group in "${groups[i]}" "${groups[i+1]}"; do
            demographic_model="$1/model_weights/demographic_model_${train_group}.pth"
            demographic_folder="${DEMOGRAPHICS_ROOT}/${test_group}"
            python3 parameterized_multiple_finger_tester.py \
                --dataset $demographic_folder \
                --weights $demographic_model \
                --cuda $cuda \
                --num_fingers $num_fingers \
                --output_root "$FAIRNESS_OUTPUT_FOLDER/sd302/train_${train_group}_test_${test_group}" \
                --scale_factor 2 \
                --diff_fingers_across_sets \
                --diff_fingers_within_set \
                --diff_sensors_across_sets \
                --same_sensor_within_set
        done
    done
done

##############################
### CONFUSION DIAGRAMS
###############
CONFUSION_DIAGRAM_FOLDER="$1/paper_results/confusion_diagram"
for ((idx=0; idx<$num_datasets; idx++))
do
    curr_dataset=${datasets[idx]}
    curr_name=${names[idx]}
    python3 parameterized_multiple_finger_tester.py \
        --dataset $curr_dataset \
        --weights $BASED_MODEL \
        --cuda "cuda:$2" \
        --num_fingers 1 \
        --output_root $CONFUSION_DIAGRAM_FOLDER/$curr_name \
        --scale_factor 1 \
        --diff_fingers_across_sets \
        --diff_fingers_within_set \
        --diff_sensors_across_sets \
        --same_sensor_within_set \
        --track_confusion_examples
done

##############################
### DATA ABLATION STUDY
###############
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