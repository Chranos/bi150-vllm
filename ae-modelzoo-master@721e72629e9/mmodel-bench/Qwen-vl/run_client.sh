MODEL_PATH=$1
#/data3t/ckpt/Qwen/Qwen2.5-VL-7B-Instruct

python3 benmark_serving_qwenvl.py \
	--url http://0.0.0.0:8000 \
	--tokenizer_path $MODEL_PATH \
	-c 32 \
	--input_num 32 \
	--input_len 30 \
	--output_len 300 \
	--trust_remote_code \
	--use_v1_api \
	--multi_modal_data 448_488.png
