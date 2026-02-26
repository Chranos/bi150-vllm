import shutil

from glob import glob
from tqdm import tqdm
from quant_utils import *
from argparse import ArgumentParser
from safetensors.torch import load_file, save_file


def main(input_path, output_path, quant_type, version):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)

    custom_ignores = ['lm_head', 'gate.weight', 'norm', 'bias', 'embed_tokens', "block_sparse_moe"]

    model_index_file = os.path.join(input_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    loaded_files = {}
    new_weight_map = {}

    for file in os.listdir(input_path):
        if not os.path.isdir(os.path.join(input_path, file)) and not file.endswith(".safetensors"):
            shutil.copy(os.path.join(input_path, file), os.path.join(output_path, file))
    
    safetensor_files = sorted(glob(os.path.join(input_path, "*.safetensors")))

    ignore = []

    for safetensor_file in tqdm(safetensor_files, desc=f"Converting to {quant_type}"):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        new_state_dict = {}

        for weight_name, weight in current_state_dict.items():

            if any(x in weight_name for x in custom_ignores):
                new_state_dict[weight_name] = weight.to(torch.bfloat16)
                new_weight_map[weight_name] = file_name
                ignore.append(weight_name.rsplit(".", 1)[0])
                continue
            
            if "expert" in weight_name:
                if quant_type == "w4a8":
                    quant_weights, scale, i8scales, i8zeros = weight_quant_moe_int4_pack8(weight)
                elif quant_type == "w8a8":
                    quant_weights, scale = weight_quant_int8(weight)
                    i8scales, i8zeros = None, None
                else:
                    raise NotImplemented

                if version == 2:
                    scale = scale.contiguous().view(1, -1)
                else:
                    assert scale.is_contiguous()

                if i8scales is not None:
                    i8scales = i8scales.squeeze_(0)
                    assert i8scales.dim() == 2
                        
                if i8zeros is not None:
                    i8zeros = i8zeros.squeeze_(0)
                    assert i8zeros.dim() == 2
                
                scale_name = weight_name + "_scale"
                i8scales_name = weight_name + "_i8_weight_scale"
                i8zeros_name = weight_name + "_i8_weight_zero"

                new_weights = quant_weights.contiguous()
                scales = scale.contiguous()

                if quant_type == "w8a8":
                    scales = scale.T.contiguous()

                if i8scales is not None:
                    new_state_dict[i8scales_name] = i8scales
                if i8zeros is not None:
                    new_state_dict[i8zeros_name] = i8zeros

                new_state_dict[weight_name] = new_weights
                new_state_dict[scale_name] = scales

                new_weight_map[weight_name] = file_name
                new_weight_map[scale_name] = file_name
                new_weight_map[i8scales_name] = file_name
                new_weight_map[i8zeros_name] = file_name
            elif weight.ndim == 2 and "weight" in weight_name:
                int8_v, scale = weight_quant_int8(weight)
                new_state_dict[weight_name] = int8_v
                new_state_dict[f"{weight_name}_scale"] = scale
                new_weight_map[weight_name] = file_name
                new_weight_map[f"{weight_name}_scale"] = file_name
            else:
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name
        
        save_file(new_state_dict, os.path.join(output_path, file_name))

        del current_state_dict
        torch.cuda.empty_cache()
    
    process_config_weight_map(input_path, output_path, ignore)
    model_index["weight_map"] = new_weight_map
    new_model_index_file = os.path.join(output_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"✅ {quant_type} 模型转换完成，结果已保存到: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--quant-type", type=str, choices=["w4a8", "w8a8"], default="w4a8")
    parser.add_argument("--version", type=int, default=2)
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.quant_type, args.version)