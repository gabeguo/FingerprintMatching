# Train on MINDTCT minutiae, SD302
python3 parameterized_runner.py \
    --datasets '/data/therealgabeguo/fingerprint_data/mindtct_minutiae/sd302' \
    --val-datasets '/data/therealgabeguo/fingerprint_data/mindtct_minutiae/sd302' \
    --posttrained-model-path /data/therealgabeguo/fingerprint_weights/mindtct_minutiae_sd302.pth \
    --diff-fingers-across-sets-train \
    --diff-sensors-across-sets-train \
    --diff-fingers-across-sets-val \
    --diff-sensors-across-sets-val \
    --scale-factor 1 \
    --log-interval 100
# Test on MINDTCT minutiae, SD302
python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/mindtct_minutiae/sd302 \
    --weights /data/therealgabeguo/fingerprint_weights/mindtct_minutiae_sd302.pth \
    --cuda 'cuda:2' \
    --num_fingers 1 \
    --output_root /data/therealgabeguo/paper_results/explaining_correlation/sd302/minutiae_mindtct \
    --scale_factor 4 \
    --diff_fingers_across_sets \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set