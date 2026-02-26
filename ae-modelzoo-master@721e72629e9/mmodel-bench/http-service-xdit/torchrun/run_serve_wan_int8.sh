

set -ex

export WORD_RANK_SUPPORT_TP=1
export ENABLE_IXFORMER_INFERENCE=1
export ENABLE_IXFORMER_SAGEATTN=1
export TOKENIZERS_PARALLELISM=true
export ENABLE_IXFORMER_W8A8LINEAR=1

python3 entrypoints/launch_host.py --config config/config_wan2.1_t2v-int8.json
# python3 entrypoints/launch_host.py --config config/config_wan2.2_t2v.json
# python3 entrypoints/launch_host.py --config config/config_wan2.2_i2v.json