# Usage: bash run_feature_correlation.sh output_directory cuda_num
# Purpose: to investigate feature-by-feature (full, binarized, orientation, frequency, minutiae) correlations

# Datasets
SD302='/data/therealgabeguo/fingerprint_data/sd302_split'
FEATURE_EXTRACTIONS_ROOT="/data/therealgabeguo/fingerprint_data/sd302_feature_extractions"
ENHANCED="enhanced"
ORIENTATION="orient"
FREQUENCY="freq"
MINDTCT_MINUTIAE_ROOT="/data/therealgabeguo/fingerprint_data/mindtct_minutiae/sd302"

######
# Training specific feature models
######
FER=$FEATURE_EXTRACTIONS_ROOT
datasets=($FER/$ENHANCED $FER/$ORIENTATION $FER/$FREQUENCY $SD302 $MINDTCT_MINUTIAE_ROOT)
nicknames=($ENHANCED $ORIENTATION $FREQUENCY "unpretrained_sd302" "minutiae")
for ((i=0; i<${#datasets[@]}; ++i));
do
    CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
        --datasets ${datasets[i]} \
        --val-datasets ${datasets[i]} \
        --posttrained-model-path $1/model_weights/feature_model_${nicknames[i]}.pth \
        --temp_model_dir 'temp_weights' --results_dir "$1/results" \
        --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
        --scale-factor 1 --log-interval 100    
done

##############################
### Testing specific feature models
###############
FEATURE_CORRELATION_FOLDER="$1/paper_results/feature_correlation/sd302"
for ((i=0; i<${#datasets[@]}; ++i));
do
    python3 parameterized_multiple_finger_tester.py \
        --dataset ${datasets[i]} \
        --weights $1/model_weights/feature_model_${nicknames[i]}.pth \
        --cuda "cuda:$2" \
        --num_fingers 1 \
        --output_root $FEATURE_CORRELATION_FOLDER/${nicknames[i]} \
        --scale_factor 4 \
        --diff_fingers_across_sets \
        --diff_fingers_within_set \
        --diff_sensors_across_sets \
        --same_sensor_within_set
done
