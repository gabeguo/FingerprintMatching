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
DEMOGRAPHICS_TEST_SPLIT_SD302='/data/verifiedanivray/sd302_test_demographic_split'
DEMOGRAPHICS_TEST_SPLIT_SD301='/data/verifiedanivray/sd301_test_demographic_split'
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
CONFIDENCE_INTERVAL_FOLDER="$1/paper_results/confidence_intervals"

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
        --output_root "${CONFIDENCE_INTERVAL_FOLDER}"/${folder_names[i]} \
        --scale_factor ${scale_factors[i]} \
        --diff_fingers_across_sets \
        --diff_fingers_within_set \
        --diff_sensors_across_sets \
        --same_sensor_within_set
done

##############################
### EXAMINE THE SPECIFIC FEATURES THAT ARE CORRELATED
###############
FEATURE_CORRELATION_FOLDER="$1/paper_results/confidence_intervals/feature_correlation/sd302"
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