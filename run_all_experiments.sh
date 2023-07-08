# variables
root_folder='/data/therealgabeguo/_reproduced_fingerprint_results'
cuda=2

## Pretrain on PrintsGAN
cd synthetic_data_transfer_learning
CUDA_VISIBLE_DEVICES=$cuda python3 runner.py \
    --model_path "{$root_folder}/model_weights/embedding_net_weights_printsgan.pth" \
    --data_path "/data/therealgabeguo/printsgan" \
    --output_folder "{$root_folder}/pretrain_results"
cd ../

## Regular train
# TODO