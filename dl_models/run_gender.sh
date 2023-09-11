# run_demographic.sh root_directory cuda

FAIRNESS_OUTPUT_FOLDER=$1/paper_results/fairness

DEMOGRAPHICS_ROOT="/home/gabeguo/data/demographics_datasets"
MALE_GROUP="sd302_male_split"
FEMALE_GROUP="sd302_female_split"

cd ../directory_organization
mkdir -p output
echo "starting demographic splitting" > output/gender_splits.txt # overwrites old stuff
for i in {0..80..20}
do
    # Shuffle
    echo "" >> ../directory_organization/output/gender_splits.txt
    cd ../directory_organization
    for subdir in $MALE_GROUP $FEMALE_GROUP
    do
        echo $subdir >> output/gender_splits.txt
        python3 split_people.py --data_folder "${DEMOGRAPHICS_ROOT}/${subdir}" \
            --train_percent 80 --val_percent 10 --rotate $i >> output/gender_splits.txt
        echo "" >> output/gender_splits.txt
    done

    cd ../dl_models

    # Train individual demographics
    for category in $MALE_GROUP $FEMALE_GROUP
    do
        folder="${DEMOGRAPHICS_ROOT}/${category}"
        CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
            --datasets "${folder}" \
            --val-datasets "${folder}" \
            --posttrained-model-path $1/model_weights/demographic_model_${category}_${i}.pth \
            --temp_model_dir 'temp_weights' --results_dir "$1/results" \
            --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
            --scale-factor 2 --log-interval 100 \
            --early-stopping-interval 105
    done
    # Also train combined demographics
    folders="${DEMOGRAPHICS_ROOT}/${MALE_GROUP} ${DEMOGRAPHICS_ROOT}/${FEMALE_GROUP}"
    CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
        --datasets "${folders}" \
        --val-datasets "${folders}" \
        --posttrained-model-path $1/model_weights/demographic_model_combined_gender_${i}.pth \
        --temp_model_dir 'temp_weights' --results_dir "$1/results" \
        --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
        --scale-factor 2 --log-interval 100 \
        --early-stopping-interval 105 

    ### Test generalizability
    for train_group in $MALE_GROUP $FEMALE_GROUP "combined_gender"
    do
        for test_group in $MALE_GROUP $FEMALE_GROUP
        do
            demographic_model=$1/model_weights/demographic_model_${train_group}_${i}.pth
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