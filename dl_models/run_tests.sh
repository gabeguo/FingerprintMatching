# # General correlation: SD300 (full)
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd300a_split/ \
#     --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
#     --cuda 'cuda:2' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/proving_correlation/general/sd300 \
#     --scale_factor 2 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set
# # General correlation: SD301 (full)
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd301_split/ \
#     --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
#     --cuda 'cuda:2' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/proving_correlation/general/sd301 \
#     --scale_factor 4 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set
# # General correlation: SD302 (full test)
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_split/ \
#     --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
#     --cuda 'cuda:2' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/proving_correlation/general/sd302 \
#     --scale_factor 4 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set
# # Finger-by-finger correlation: SD300 (balanced)
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd300a_split/ \
#     --weights /data/therealgabeguo/fingerprint_weights/balanced_model_sd302_sd300.pth \
#     --cuda 'cuda:1' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/proving_correlation/fingerByfinger_new/sd300 \
#     --scale_factor 2 \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set
# # Finger-by-finger correlation: SD301 (balanced)
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd301_split_balanced/ \
#     --weights /data/therealgabeguo/fingerprint_weights/balanced_model_sd302_sd300.pth \
#     --cuda 'cuda:1' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/proving_correlation/fingerByfinger_new/sd301 \
#     --scale_factor 4 \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set
# # Finger-by-finger correlation: SD302 (balanced)
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_split_balanced/ \
#     --weights /data/therealgabeguo/fingerprint_weights/balanced_model_sd302_sd300.pth \
#     --cuda 'cuda:1' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/proving_correlation/fingerByfinger_new/sd302 \
#     --scale_factor 4 \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set
# # Specific correlation: SD302 (un-pretrained)
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_split \
#     --weights /data/therealgabeguo/fingerprint_weights/embedding_net_weights_sd302_split.pth \
#     --cuda 'cuda:2' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/explaining_correlation/sd302/baseline_noPretrain \
#     --scale_factor 4 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set
# # Specific correlation: SD302 (enhanced)
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/enhanced \
#     --weights /data/therealgabeguo/fingerprint_weights/embedding_net_weights_enhanced.pth \
#     --cuda 'cuda:2' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/explaining_correlation/sd302/enhanced \
#     --scale_factor 4 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set
# # Specific correlation: SD302 (minutiae)
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/minutiae \
#     --weights /data/therealgabeguo/fingerprint_weights/embedding_net_weights_minutiae.pth \
#     --cuda 'cuda:2' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/explaining_correlation/sd302/minutiae \
#     --scale_factor 4 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set
# # Specific correlation: SD302 (orient)
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/orient \
#     --weights /data/therealgabeguo/fingerprint_weights/embedding_net_weights_orient.pth \
#     --cuda 'cuda:2' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/explaining_correlation/sd302/orient \
#     --scale_factor 4 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set
# # Specific correlation: SD302 (freq)
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/freq \
#     --weights /data/therealgabeguo/fingerprint_weights/embedding_net_weights_freq.pth \
#     --cuda 'cuda:2' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/explaining_correlation/sd302/freq \
#     --scale_factor 4 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set

# for j in 1 2 3
# do
#     for i in 1 2 3 4 5
#     do
        # python3 parameterized_multiple_finger_tester.py \
        #     --dataset /data/therealgabeguo/fingerprint_data/sd302_split_balanced/ \
        #     --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
        #     --cuda 'cuda:2' \
        #     --num_fingers $i \
        #     --output_root /data/therealgabeguo/paper_results/multi_finger/sd302/${i} \
        #     --scale_factor 1 \
        #     --diff_fingers_across_sets \
        #     --diff_fingers_within_set \
        #     --diff_sensors_across_sets \
        #     --same_sensor_within_set
        # python3 parameterized_multiple_finger_tester.py \
        #     --dataset /data/therealgabeguo/fingerprint_data/sd301_split_balanced/ \
        #     --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
        #     --cuda 'cuda:2' \
        #     --num_fingers $i \
        #     --output_root /data/therealgabeguo/paper_results/multi_finger/sd301/${i} \
        #     --scale_factor 1 \
        #     --diff_fingers_across_sets \
        #     --diff_fingers_within_set \
        #     --diff_sensors_across_sets \
        #     --same_sensor_within_set
        # python3 parameterized_multiple_finger_tester.py \
        #     --dataset /data/therealgabeguo/fingerprint_data/sd300a_split_balanced/ \
        #     --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
        #     --cuda 'cuda:2' \
        #     --num_fingers $i \
        #     --output_root /data/therealgabeguo/paper_results/multi_finger/sd300/${i} \
        #     --scale_factor 1 \
        #     --diff_fingers_across_sets \
        #     --diff_fingers_within_set \
        #     --diff_sensors_across_sets \
        #     --same_sensor_within_set
#     done
# done
# # Confusion diagram: SD300
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd300a_split/ \
#     --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
#     --cuda 'cuda:1' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/confusion_diagram/sd300 \
#     --scale_factor 1 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set \
#     --track_confusion_examples
# # Confusion diagram: SD301
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd301_split/ \
#     --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
#     --cuda 'cuda:1' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/confusion_diagram/sd301 \
#     --scale_factor 1 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set \
#     --track_confusion_examples
# Confusion diagram: SD302
# # python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_split/ \
#     --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
#     --cuda 'cuda:1' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/confusion_diagram/sd302 \
#     --scale_factor 1 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set \
#     --track_confusion_examples
# # Generalizing models trained on some fingers to other fingers
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_split/ \
#     --weights /data/therealgabeguo/fingerprint_weights/dual_02_03_same_finger_allowed_model.pth \
#     --cuda 'cuda:1' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/generalizing_sub_model/sd302 \
#     --scale_factor 4 \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set \
# # High-quality images: NFIQ score == 1, 2
# python3 -W "ignore" parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_high_quality/ \
#     --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
#     --cuda 'cuda:2' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/high_quality/sd302 \
#     --scale_factor 2 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set






# # Two-finger (only different) prediction
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_split/ \
#     --weights /data/therealgabeguo/fingerprint_weights/dual_02_03_model.pth \
#     --cuda 'cuda:1' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/results \
#     --scale_factor 4 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set \
#     --possible_fgrps "02 03"
# # Two-finger (same allowed) prediction
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_split/ \
#     --weights /data/therealgabeguo/fingerprint_weights/dual_02_03_same_finger_allowed_model.pth \
#     --cuda 'cuda:1' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/results \
#     --scale_factor 4 \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set \
#     --possible_fgrps "02 03"
# # Generalizing models trained on some fingers to other fingers
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_split/ \
#     --weights /data/therealgabeguo/fingerprint_weights/dual_02_03_same_finger_allowed_model.pth \
#     --cuda 'cuda:1' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/results \
#     --scale_factor 4 \
#     --diff_fingers_across_sets \
#     --diff_fingers_within_set \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set \
#     --possible_fgrps "04 05"
# # Transfer learning on two fingers
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_split/ \
#     --weights /data/therealgabeguo/fingerprint_weights/self_02_model.pth \
#     --cuda 'cuda:1' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/results \
#     --scale_factor 4 \
#     --diff_sensors_across_sets \
#     --possible_fgrps "02 03"
# # Testing same-finger correlation on full model when all fingers allowed: SD302
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd302_split/ \
#     --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
#     --cuda 'cuda:2' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/proving_correlation/same_allowed/sd302 \
#     --scale_factor 4 \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set
# # Testing same-finger correlation on full model when all fingers allowed: SD301
# python3 parameterized_multiple_finger_tester.py \
#     --dataset /data/therealgabeguo/fingerprint_data/sd301_split/ \
#     --weights /data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11\:06\:28.pth \
#     --cuda 'cuda:2' \
#     --num_fingers 1 \
#     --output_root /data/therealgabeguo/paper_results/proving_correlation/same_allowed/sd301 \
#     --scale_factor 4 \
#     --diff_sensors_across_sets \
#     --same_sensor_within_set



# Possible other tests:
# SD301 feature extractions
# un-pretrained SD302