python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd300a_split \
    --weights /data/therealgabeguo/updated_fingerprint_results_fall23/model_weights/full_based_model_PRETRAINED.pth \
    --output_root /data/therealgabeguo/updated_fingerprint_results_fall23/gradcam_outputs_fingerprint/sd300

python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd301_split \
    --weights /data/therealgabeguo/updated_fingerprint_results_fall23/model_weights/full_based_model_PRETRAINED.pth \
    --output_root /data/therealgabeguo/updated_fingerprint_results_fall23/gradcam_outputs_fingerprint/sd301

python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_split \
    --weights /data/therealgabeguo/updated_fingerprint_results_fall23/model_weights/full_based_model_PRETRAINED.pth \
    --output_root /data/therealgabeguo/updated_fingerprint_results_fall23/gradcam_outputs_fingerprint/sd302

python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/enhanced \
    --weights /data/therealgabeguo/updated_fingerprint_results_fall23/model_weights/feature_model_enhanced.pth \
    --output_root /data/therealgabeguo/updated_fingerprint_results_fall23/gradcam_outputs_fingerprint/sd302_enhanced

python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/freq \
    --weights /data/therealgabeguo/updated_fingerprint_results_fall23/model_weights/feature_model_freq.pth \
    --output_root /data/therealgabeguo/updated_fingerprint_results_fall23/gradcam_outputs_fingerprint/sd302_freq

python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/orient \
    --weights /data/therealgabeguo/updated_fingerprint_results_fall23/model_weights/feature_model_orient.pth \
    --output_root /data/therealgabeguo/updated_fingerprint_results_fall23/gradcam_outputs_fingerprint/sd302_orient

python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/mindtct_minutiae/sd302 \
    --weights /data/therealgabeguo/updated_fingerprint_results_fall23/model_weights/feature_model_minutiae.pth \
    --output_root /data/therealgabeguo/updated_fingerprint_results_fall23/gradcam_outputs_fingerprint/sd302_minutiae