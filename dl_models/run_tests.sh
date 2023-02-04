# General correlation: SD301 (full)
python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd301_split/ \
    --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
    --cuda 'cuda:2' \
    --num_fingers 1 \
    --output_root /data/therealgabeguo/paper_results/proving_correlation/general/sd301 \
    --scale_factor 4 \
    --diff_fingers_across_sets \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set
# General correlation: SD302 (full test)
python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_split/ \
    --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
    --cuda 'cuda:2' \
    --num_fingers 1 \
    --output_root /data/therealgabeguo/paper_results/proving_correlation/general/sd302 \
    --scale_factor 4 \
    --diff_fingers_across_sets \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set
# Finger-by-finger correlation: SD301 (balanced)
python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd301_split_balanced/ \
    --weights /data/therealgabeguo/most_recent_experiment_reports/fingerprint_weights/embedding_net_weights_sd302_split_balanced.pth \
    --cuda 'cuda:2' \
    --num_fingers 1 \
    --output_root /data/therealgabeguo/paper_results/proving_correlation/fingerByfinger/sd301 \
    --scale_factor 4 \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set
# Finger-by-finger correlation: SD302 (balanced)
python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_split_balanced/ \
    --weights /data/therealgabeguo/most_recent_experiment_reports/fingerprint_weights/embedding_net_weights_sd302_split_balanced.pth \
    --cuda 'cuda:2' \
    --num_fingers 1 \
    --output_root /data/therealgabeguo/paper_results/proving_correlation/fingerByfinger/sd302 \
    --scale_factor 4 \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set
# Specific correlation: SD302 (enhanced)
python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/enhanced \
    --weights /data/therealgabeguo/most_recent_experiment_reports/fingerprint_weights/embedding_net_weights_enhanced.pth \
    --cuda 'cuda:2' \
    --num_fingers 1 \
    --output_root /data/therealgabeguo/paper_results/explaining_correlation/sd302/enhanced \
    --scale_factor 4 \
    --diff_fingers_across_sets \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set
# Specific correlation: SD302 (minutiae)
python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/minutiae \
    --weights /data/therealgabeguo/most_recent_experiment_reports/fingerprint_weights/embedding_net_weights_minutiae.pth \
    --cuda 'cuda:2' \
    --num_fingers 1 \
    --output_root /data/therealgabeguo/paper_results/explaining_correlation/sd302/minutiae \
    --scale_factor 4 \
    --diff_fingers_across_sets \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set
# Specific correlation: SD302 (orient)
python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/orient \
    --weights /data/therealgabeguo/most_recent_experiment_reports/fingerprint_weights/embedding_net_weights_orient.pth \
    --cuda 'cuda:2' \
    --num_fingers 1 \
    --output_root /data/therealgabeguo/paper_results/explaining_correlation/sd302/orient \
    --scale_factor 4 \
    --diff_fingers_across_sets \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set
# Specific correlation: SD302 (freq)
python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/freq \
    --weights /data/therealgabeguo/most_recent_experiment_reports/fingerprint_weights/embedding_net_weights_freq.pth \
    --cuda 'cuda:2' \
    --num_fingers 1 \
    --output_root /data/therealgabeguo/paper_results/explaining_correlation/sd302/freq \
    --scale_factor 4 \
    --diff_fingers_across_sets \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set

for i in 1 2 3 4 5
do
   python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_split/ \
    --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
    --cuda 'cuda:2' \
    --num_fingers $i \
    --output_root /data/therealgabeguo/paper_results/multi_finger/sd302/${i} \
    --scale_factor 1 \
    --diff_fingers_across_sets \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set
   python3 parameterized_multiple_finger_tester.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd301_split/ \
    --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
    --cuda 'cuda:2' \
    --num_fingers $i \
    --output_root /data/therealgabeguo/paper_results/multi_finger/sd301/${i} \
    --scale_factor 1 \
    --diff_fingers_across_sets \
    --diff_fingers_within_set \
    --diff_sensors_across_sets \
    --same_sensor_within_set
done

# Required other tests:
# Lead analysis

# Possible other tests:
# SD301 feature extractions
# un-pretrained SD302