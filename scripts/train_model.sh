cd ..
CUDA_VISIBLE_DEVICES=5 python transformer_discrete_main.py \
--cuda \
--pose3d-folder="/data/jiaman/github/cvpr20_dance/json_data/block_20s_train_val_data_discrete300/block_20s_pose3d_npy_files" \
--discrete-pose3d-folder="/data/jiaman/github/cvpr20_dance/json_data/block_20s_train_val_data_discrete300/block_20s_pose3d_discrete_npy_files" \
--train-json-file="/data/jiaman/github/cvpr20_dance/json_data/block_20s_train_val_data_discrete300/block_20s_train_pose3d_data.json" \
--val-json-file="/data/jiaman/github/cvpr20_dance/json_data/block_20s_train_val_data_discrete300/block_20s_val_pose3d_data.json" \
--train-discrete-json-file="/data/jiaman/github/cvpr20_dance/json_data/block_20s_train_val_data_discrete300/block_20s_train_pose3d_data.json" \
--val-discrete-json-file="/data/jiaman/github/cvpr20_dance/json_data/block_20s_train_val_data_discrete300/block_20s_val_pose3d_data.json" \
--checkpoint-folder="checkpoint/block_20s_discrete300_transformer/lr1e-4_bs32_feats10_dout10" \
--log-dir="tensorboard/block_20s_discrete300_transformer/lr1e-4_bs32_feats10_dout10" \
--batch-size=32 \
--lr=1e-4 \
--n-dec-layers=4 \
--n-head=4 \
--d-model=128 \
--d-k=128 \
--d-v=128 \
--epochs=2000 \
--lr-steps 800 1600 \
--max-timesteps=480 \
--num-cls=300 \
--feats-dim=10 \
--d-out=10 \
2>&1 |tee logs/train_discrete300_transformer_decoder_block_20s_lr1e-4_bs32_feats10_dout10.log

