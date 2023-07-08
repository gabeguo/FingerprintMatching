# sh train.sh root_folder cuda

##########
# Official training
##########

SD302='/data/therealgabeguo/fingerprint_data/sd302_split'
SD301='/data/therealgabeguo/fingerprint_data/sd301_split'
SD300='/data/therealgabeguo/fingerprint_data/sd300a_split'
RIDGEBASE='/data/therealgabeguo/fingerprint_data/RidgeBase_Split'
SD302_BALANCED='/data/therealgabeguo/fingerprint_data/sd302_split_balanced'

DEMOGRAPHICS_ROOT="/data/verifiedanivray/demographics_datasets"
CAUCASIAN_DESCENT="sd302_white_split"
NON_CAUCASIAN="sd302_non-white_split"
MALE_GROUP="sd302_male_split"
FEMALE_GROUP="sd302_female_split"

FEATURE_EXTRACTIONS_ROOT="/data/therealgabeguo/fingerprint_data/sd302_feature_extractions"
ENHANCED="enhanced"
ORIENTATION="orient"
FREQUENCY="freq"

MINDTCT_MINUTIAE_ROOT="/data/therealgabeguo/fingerprint_data/mindtct_minutiae/sd302"

######
# Training base model
######
CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
    --datasets "${SD302} ${SD300} ${RIDGEBASE}" \
    --val-datasets "${SD302} ${SD300} ${RIDGEBASE}" \
    --pretrained-model-path $1/model_weights/embedding_net_weights_printsgan.pth \
    --posttrained-model-path $1/model_weights/based_model.pth \
    --temp_model_dir 'temp_weights' --results_dir "$1/results" \
    --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
    --scale-factor 2 --log-interval 100
######
# Training balanced model
######
CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
    --datasets "${SD302_BALANCED} ${SD300}" \
    --val-datasets "${SD302_BALANCED} ${SD300}" \
    --posttrained-model-path $1/model_weights/balanced_model_sd302_sd300.pth \
    --temp_model_dir 'temp_weights' --results_dir "$1/results" \
    --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
    --scale-factor 1 --log-interval 100
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
######
# Training specific feature models
######
for feature in $ENHANCED $ORIENTATION $FREQUENCY
do
    folder="${FEATURE_EXTRACTIONS_ROOT}/${feature}"
    CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
        --datasets "${folder}" \
        --val-datasets "${folder}" \
        --posttrained-model-path $1/model_weights/feature_model_${feature}.pth \
        --temp_model_dir 'temp_weights' --results_dir "$1/results" \
        --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
        --scale-factor 1 --log-interval 100    
done
# Training un-pre-trained SD302 model, for comparison with feature-specific
CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
    --datasets "${SD302}" \
    --val-datasets "${SD302}" \
    --posttrained-model-path $1/model_weights/unpretrained_model_sd302.pth \
    --temp_model_dir 'temp_weights' --results_dir "$1/results" \
    --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
    --scale-factor 2 --log-interval 100
# minutiae needs a separate one, because it was created with MINDTCT
CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
    --datasets $MINDTCT_MINUTIAE_ROOT \
    --val-datasets $MINDTCT_MINUTIAE_ROOT \
    --posttrained-model-path $1/model_weights/feature_model_minutiae.pth \
    --temp_model_dir 'temp_weights' --results_dir "$1/results" \
    --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
    --scale-factor 1 --log-interval 100    
######
# Two-finger experiment
######
CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
    --datasets "${SD302} ${SD300} ${RIDGEBASE}" \
    --val-datasets "${SD302}" \
    --possible-fgrps '02 03' \
    --posttrained-model-path $1/model_weights/dual_02_03.pth \
    --temp_model_dir 'temp_weights' \
    --results_dir "$1/results" \
    --diff-sensors-across-sets-val \
    --scale-factor 4 \
    --log-interval 20
