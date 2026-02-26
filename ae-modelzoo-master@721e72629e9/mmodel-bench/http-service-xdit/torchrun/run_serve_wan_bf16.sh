

set -ex

export WORD_RANK_SUPPORT_TP=1
export ENABLE_IXFORMER_INFERENCE=1
export ENABLE_IXFORMER_SAGEATTN=1
export TOKENIZERS_PARALLELISM=true


# python3 entrypoints/launch_host.py --config config/config_wan2.1_t2v-bf16.json
python3 entrypoints/launch_host.py --config config/config_wan2.2_t2v.json
# python3 entrypoints/launch_host.py --config config/config_wan2.2_i2v.json