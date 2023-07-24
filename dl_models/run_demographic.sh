# run_demographic.sh root_directory cuda

FAIRNESS_OUTPUT_FOLDER=$1/paper_results/fairness

DEMOGRAPHICS_ROOT="/data/therealgabeguo/demographics_datasets"
CAUCASIAN_DESCENT="sd302_white_split"
NON_CAUCASIAN="sd302_non-white_split"
MALE_GROUP="sd302_male_split"
FEMALE_GROUP="sd302_female_split"

cd ../directory_organization
mkdir -p output
echo "starting demographic splitting" > output/splits.txt # overwrites old stuff
for i in {0..9}
do
    echo "\n" >> ../directory_organization/output/splits.txt
    cd ../directory_organization
    for subdir in $CAUCASIAN_DESCENT $NON_CAUCASIAN $MALE_GROUP $FEMALE_GROUP
    do
        echo $subdir >> output/splits.txt
        python3 split_people.py --data_folder "${DEMOGRAPHICS_ROOT}/${subdir}" \
            --train_percent 80 --val_percent 10 --rotate 10 >> output/splits.txt
        echo "\n" >> output/splits.txt
    done

    cd ../dl_models

    ######
    # Training demographic models
    ######
    # Train individual demographics
    for category in $CAUCASIAN_DESCENT $NON_CAUCASIAN $MALE_GROUP $FEMALE_GROUP
    do
        folder="${DEMOGRAPHICS_ROOT}/${category}"
        CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
            --datasets "${folder}" \
            --val-datasets "${folder}" \
            --posttrained-model-path $1/model_weights/demographic_model_${category}.pth \
            --temp_model_dir 'temp_weights' --results_dir "$1/results" \
            --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
            --scale-factor 2 --log-interval 100 --early-stopping-interval 25
    done
    # Also train combined demographics
    folders="${DEMOGRAPHICS_ROOT}/${CAUCASIAN_DESCENT} ${DEMOGRAPHICS_ROOT}/${NON_CAUCASIAN}"
    CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
        --datasets "${folders}" \
        --val-datasets "${folders}" \
        --posttrained-model-path $1/model_weights/demographic_model_combined_race.pth \
        --temp_model_dir 'temp_weights' --results_dir "$1/results" \
        --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
        --scale-factor 2 --log-interval 100 --early-stopping-interval 25
    folders="${DEMOGRAPHICS_ROOT}/${MALE_GROUP} ${DEMOGRAPHICS_ROOT}/${FEMALE_GROUP}"
    CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
        --datasets "${folders}" \
        --val-datasets "${folders}" \
        --posttrained-model-path $1/model_weights/demographic_model_combined_gender.pth \
        --temp_model_dir 'temp_weights' --results_dir "$1/results" \
        --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
        --scale-factor 2 --log-interval 100 --early-stopping-interval 25     

    #########
    # Tests generalizability
    #########################

    ### Race
    for train_group in $CAUCASIAN_DESCENT $NON_CAUCASIAN "combined_race"
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
                --scale_factor 4 \
                --diff_fingers_across_sets \
                --diff_fingers_within_set \
                --diff_sensors_across_sets \
                --same_sensor_within_set
        done
    done
    ### Gender
    for train_group in $MALE_GROUP $FEMALE_GROUP "combined_gender"
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
                --scale_factor 4 \
                --diff_fingers_across_sets \
                --diff_fingers_within_set \
                --diff_sensors_across_sets \
                --same_sensor_within_set
        done
    done
done

# #######
# # Tests fairness
# #################

# ORIG_TEST_SET_ROOT='/data/verifiedanivray/sd302_test_demographic_split'

# for group_name in 'female' 'male' 'non-white' 'white'
# do
#     full_model=/data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11:06:28.pth
#     python3 parameterized_multiple_finger_tester.py \
#         --dataset $ORIG_TEST_SET_ROOT/$group_name \
#         --weights $full_model \
#         --cuda "cuda:$2" \
#         --num_fingers 1 \
#         --output_root "$FAIRNESS_OUTPUT_FOLDER/full_model_sd302test/test_${group_name}" \
#         --scale_factor 4 \
#         --diff_fingers_across_sets \
#         --diff_fingers_within_set \
#         --diff_sensors_across_sets \
#         --same_sensor_within_set
# done

# ORIG_TEST_SET_ROOT='/data/verifiedanivray/sd301_test_demographic_split'

# for group_name in 'Female' 'Male'
# do
#     full_model=/data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11:06:28.pth
#     python3 parameterized_multiple_finger_tester.py \
#         --dataset $ORIG_TEST_SET_ROOT/$group_name \
#         --weights $full_model \
#         --cuda "cuda:$2" \
#         --num_fingers 1 \
#         --output_root "$FAIRNESS_OUTPUT_FOLDER/full_model_sd301test/test_${group_name}" \
#         --scale_factor 4 \
#         --diff_fingers_across_sets \
#         --diff_fingers_within_set \
#         --diff_sensors_across_sets \
#         --same_sensor_within_set
# done