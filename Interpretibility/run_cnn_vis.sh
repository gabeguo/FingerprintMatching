#!/bin/bash
source ../venv/bin/activate

python cnn_layer_visualization.py \
    --model_path /data/therealgabeguo/updated_fingerprint_results_fall23/model_weights/full_based_model_PRETRAINED.pth \
    --output_folder /data/verifiedanivray/generated_early2 \
    --conv_layer "4.1.conv2"

python cnn_layer_visualization.py \
    --model_path /data/therealgabeguo/updated_fingerprint_results_fall23/model_weights/full_based_model_PRETRAINED.pth \
    --output_folder /data/verifiedanivray/generated_middle2 \
    --conv_layer "6.0.conv2"

python cnn_layer_visualization.py \
    --model_path /data/therealgabeguo/updated_fingerprint_results_fall23/model_weights/full_based_model_PRETRAINED.pth \
    --output_folder /data/verifiedanivray/generated_end2 \
    --conv_layer "7.1.conv2"