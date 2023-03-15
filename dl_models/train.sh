# Prelim two-finger experiment
python3 parameterized_runner.py \
    --datasets '/data/therealgabeguo/fingerprint_data/sd302_split /data/therealgabeguo/fingerprint_data/RidgeBase_Split' \
    --possible-fgrps '02 03' \
    --posttrained-model-path /data/therealgabeguo/fingerprint_weights/dual_02_03_model.pth \
    --diff-fingers-across-sets-train \
    --diff-sensors-across-sets-train \
    --diff-fingers-across-sets-val \
    --diff-sensors-across-sets-val