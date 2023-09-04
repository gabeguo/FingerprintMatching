python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd300a_split \
    --output_root /data/therealgabeguo/gradcam_outputs_fingerprint/sd300

python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd301_split \
    --output_root /data/therealgabeguo/gradcam_outputs_fingerprint/sd301

python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_split \
    --output_root /data/therealgabeguo/gradcam_outputs_fingerprint/sd302

python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/enhanced \
    --weights /data/therealgabeguo/fingerprint_weights/embedding_net_weights_enhanced.pth \
    --output_root /data/therealgabeguo/gradcam_outputs_fingerprint/sd302_enhanced

python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/freq \
    --weights /data/therealgabeguo/fingerprint_weights/embedding_net_weights_freq.pth \
    --output_root /data/therealgabeguo/gradcam_outputs_fingerprint/sd302_freq

python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/sd302_feature_extractions/orient \
    --weights /data/therealgabeguo/fingerprint_weights/embedding_net_weights_orient.pth \
    --output_root /data/therealgabeguo/gradcam_outputs_fingerprint/sd302_orient

python embedding_grad_cam.py \
    --dataset /data/therealgabeguo/fingerprint_data/mindtct_minutiae/sd302 \
    --weights /data/therealgabeguo/fingerprint_weights/mindtct_minutiae_sd302.pth \
    --output_root /data/therealgabeguo/gradcam_outputs_fingerprint/sd302_minutiae