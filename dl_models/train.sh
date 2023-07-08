# train.sh root_folder cuda

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

# Training base model
CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
    --datasets "${SD302} ${SD300} ${RIDGEBASE}" \
    --val-datasets "${SD302} ${SD300} ${RIDGEBASE}" \
    --pretrained-model-path $1/model_weights/embedding_net_weights_printsgan.pth \
    --posttrained-model-path $1/model_weights/based_model.pth \
    --temp_model_dir 'temp_weights' --results_dir "$1/results" \
    --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
    --scale-factor 2 --log-interval 100
# Training balanced model
CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
    --datasets "${SD302_BALANCED} ${SD300}" \
    --val-datasets "${SD302_BALANCED} ${SD300}" \
    --posttrained-model-path $1/model_weights/balanced_model_sd302_sd300.pth \
    --temp_model_dir 'temp_weights' --results_dir "$1/results" \
    --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
    --scale-factor 1 --log-interval 100
for category in $CAUCASIAN_DESCENT $NON_CAUCASIAN $MALE_GROUP $FEMALE_GROUP
do
    folder="${DEMOGRAPHICS_ROOT}/${category}"
    CUDA_VISIBLE_DEVICES=$2 python3 parameterized_runner.py \
        --datasets "${folder}" \
        --val-datasets "${folder}" \
        --posttrained-model-path $1/model_weights/demographic_model${category}.pth \
        --temp_model_dir 'temp_weights' --results_dir "$1/results" \
        --diff-fingers-across-sets-train --diff-sensors-across-sets-train --diff-fingers-across-sets-val --diff-sensors-across-sets-val \
        --scale-factor 1 --log-interval 100
done
# Two-finger experiment
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
