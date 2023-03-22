##########
# Official training
##########
# Training balanced model
python3 parameterized_runner.py \
    --datasets '/data/therealgabeguo/fingerprint_data/sd302_split_balanced /data/therealgabeguo/fingerprint_data/sd300a_split' \
    --val-datasets '/data/therealgabeguo/fingerprint_data/sd302_split_balanced /data/therealgabeguo/fingerprint_data/sd300a_split' \
    --posttrained-model-path /data/therealgabeguo/fingerprint_weights/balanced_model_sd302_sd300.pth \
    --diff-fingers-across-sets-train \
    --diff-sensors-across-sets-train \
    --diff-fingers-across-sets-val \
    --diff-sensors-across-sets-val \
    --scale-factor 1 \
    --log-interval 100
# Two-finger (but allowing same) experiment
python3 parameterized_runner.py \
    --datasets '/data/therealgabeguo/fingerprint_data/sd302_split /data/therealgabeguo/fingerprint_data/sd300a_split /data/therealgabeguo/fingerprint_data/RidgeBase_Split' \
    --val-datasets '/data/therealgabeguo/fingerprint_data/sd302_split' \
    --possible-fgrps '02 03' \
    --posttrained-model-path /data/therealgabeguo/fingerprint_weights/dual_02_03_same_finger_allowed_model.pth \
    --diff-sensors-across-sets-val \
    --scale-factor 4 \
    --log-interval 20












##########
# Miscellaneous experiments
##########

# # Two-finger experiment
# python3 parameterized_runner.py \
#     --datasets '/data/therealgabeguo/fingerprint_data/sd302_split /data/therealgabeguo/fingerprint_data/sd300a_split /data/therealgabeguo/fingerprint_data/RidgeBase_Split' \
#     --val-datasets '/data/therealgabeguo/fingerprint_data/sd302_split' \
#     --possible-fgrps '02 03' \
#     --posttrained-model-path /data/therealgabeguo/fingerprint_weights/dual_02_03_model.pth \
#     --diff-fingers-across-sets-train \
#     --diff-fingers-across-sets-val \
#     --diff-sensors-across-sets-val \
#     --scale-factor 3 \
#     --log-interval 20
# # Same-finger experiment with pretrained weights
# python3 parameterized_runner.py \
#     --datasets '/data/therealgabeguo/fingerprint_data/sd302_split /data/therealgabeguo/fingerprint_data/sd300a_split /data/therealgabeguo/fingerprint_data/RidgeBase_Split' \
#     --val-datasets '/data/therealgabeguo/fingerprint_data/sd302_split' \
#     --possible-fgrps '02 03' \
#     --pretrained-model-path /data/therealgabeguo/embedding_net_weights_printsgan.pth \
#     --posttrained-model-path /data/therealgabeguo/fingerprint_weights/self_02_model.pth \
#     --diff-sensors-across-sets-val \
#     --scale-factor 4 \
#     --log-interval 20