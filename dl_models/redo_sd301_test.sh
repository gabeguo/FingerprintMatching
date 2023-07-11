# bash redo_sd301_test.sh cuda

# General correlation: SD301 (full)
python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd301_split/ \
    --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
    --cuda "cuda:$1" \
    --num_fingers 1 \
    --output_root /data/therealgabeguo/paper_results/proving_correlation/general/sd301_UPDATED \
    --scale_factor 4 \
    --diff_fingers_across_sets \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set

# Finger-by-finger correlation: SD301 (balanced)
python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd301_split_balanced/ \
    --weights /data/therealgabeguo/fingerprint_weights/balanced_model_sd302_sd300.pth \
    --cuda "cuda:$1" \
    --num_fingers 1 \
    --output_root /data/therealgabeguo/paper_results/proving_correlation/fingerByfinger_new/sd301_UPDATED \
    --scale_factor 4 \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set

# Multiple finger test
for j in 1 2 3
do
    for i in 1 2 3 4 5
    do
        python3 parameterized_multiple_finger_tester.py \
            --dataset /data/therealgabeguo/fingerprint_data/sd301_split_balanced/ \
            --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
            --cuda "cuda:$1" \
            --num_fingers $i \
            --output_root /data/therealgabeguo/paper_results/multi_finger/sd301_UPDATED/${i} \
            --scale_factor 1 \
            --diff_fingers_across_sets \
            --diff_fingers_within_set \
            --diff_sensors_across_sets \
            --same_sensor_within_set
    done

# Confusion diagram: SD301
python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd301_split/ \
    --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
    --cuda "cuda:$1" \
    --num_fingers 1 \
    --output_root /data/therealgabeguo/paper_results/confusion_diagram/sd301_UPDATED \
    --scale_factor 1 \
    --diff_fingers_across_sets \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set \
    --track_confusion_examples