# Prelim two-finger experiment
python3 parameterized_runner.py \
    --datasets '/data/therealgabeguo/fingerprint_data/sd302_split /data/therealgabeguo/fingerprint_data/sd300a_split /data/therealgabeguo/fingerprint_data/RidgeBase_Split' \
    --val-datasets '/data/therealgabeguo/fingerprint_data/sd302_split' \
    --possible-fgrps '02 03' \
    --posttrained-model-path /data/therealgabeguo/fingerprint_weights/dual_02_03_model.pth \
    --diff-fingers-across-sets-train \
    --diff-fingers-across-sets-val \
    --diff-sensors-across-sets-val \
    --scale-factor 3 \
    --log-interval 10