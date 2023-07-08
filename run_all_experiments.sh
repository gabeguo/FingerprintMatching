cd synthetic_data_transfer_learning
CUDA_VISIBLE_DEVICES=2 python3 runner.py \
    --model_path '/data/therealgabeguo/embedding_net_weights_printsgan.pth' \
    --data_path '/data/therealgabeguo/printsgan' \
    --output_folder '/data/therealgabeguo/pretrain_results'
cd ../