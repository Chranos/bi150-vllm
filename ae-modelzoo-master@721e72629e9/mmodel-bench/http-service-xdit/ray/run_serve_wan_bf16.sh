
export WORD_RANK_SUPPORT_TP=1

MODEL_PATH=/data1/Wan2.1-T2V-14B-Diffusers
# MODEL_PATH=/data1/Wan2.2-T2V-A14B-Diffusers
# MODEL_PATH=/data1/Wan2.2-I2V-A14B-Diffusers

python3 entrypoints/launch_wan.py --world_size 16 \
--tensor_parallel_degree 4 \
--ulysses_parallel_degree 2 \
--use_cfg_parallel \
--model_path $MODEL_PATH