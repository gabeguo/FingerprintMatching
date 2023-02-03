
python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd300a_split_balanced/ \
    --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
    --cuda 'cuda:2' \
    --num_fingers 1 \
    --output_root /data/therealgabeguo/results/ \
    --scale_factor 1 \
    --exclude_same_finger