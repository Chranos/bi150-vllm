import torch

from collections import defaultdict
from safetensors.torch import load_file

from .processor import quant_post_process
from .specs import QuantBatch, QuantBatchArtifacts, QuantSpec
from .utils import compile_ignore as compile_pattern, should_ignore, should_quant, get_format
from core.kernels import weight_quant_bf16_to_int8, weight_quant_bf16_to_int4pack8


def quant(
    file_name: str | None = None,
    state_dict: dict[str, torch.Tensor] | None = None,
    custom_quant_config: dict | None = None,  # from model.yaml
    device: torch.device = "cpu",
    model_type: str | None = None
):
    if file_name:
        assert state_dict is None
        state_dict = load_file(file_name)
    else:
        assert state_dict is not None
    
    exact_ignore, regex_ignore = compile_pattern(custom_quant_config['ignore'] or [])

    int4_exact_pattern, int4_regex_pattern = compile_pattern(custom_quant_config['class']['int4'])
    int8_exact_pattern, int8_regex_pattern = compile_pattern(custom_quant_config['class']['int8'])

    batch_size = custom_quant_config['batch_size']

    buffer = defaultdict(list)  # key = (quant_type, shape, symmetric, version, format)
    outputs = {}
    quant_map = {}

    for weight_name, weight in state_dict.items():
        # 不量化
        if should_ignore(weight_name, exact_ignore, regex_ignore):
            outputs[weight_name] = weight.cpu()
            quant_map[weight_name] = str(weight.dtype)
            continue
        
        if should_quant(weight_name, int4_exact_pattern, int4_regex_pattern):
            qs = QuantSpec(
                quant_type="int4",
                shape=tuple(weight.shape),
                symmetric=custom_quant_config['symmetric'],
                version=custom_quant_config['version'],
                format=get_format(weight_name, custom_quant_config['format'])
            )
            quant_map[weight_name] = f"int4-{get_format(weight_name, custom_quant_config['format'])}"
        elif should_quant(weight_name, int8_exact_pattern, int8_regex_pattern):
            qs = QuantSpec(
                quant_type="int8",
                shape=tuple(weight.shape),
                symmetric=custom_quant_config['symmetric'],
                version=custom_quant_config['version'],
                format=get_format(weight_name, custom_quant_config['format'])
            )
            quant_map[weight_name] = f"int8-{get_format(weight_name, custom_quant_config['format'])}"
        else:
            raise ValueError(f"{weight_name} missing in ignore or quant pattern!!!")

        buffer[qs].append(weight_name) 

    # 构建batch
    for qs, weight_names in buffer.items():
        for i in range(0, len(weight_names), batch_size):
            chunk_names = weight_names[i:i+batch_size]
            tensors = []
            for name in chunk_names:
                w = state_dict[name]
                if w.device != device:
                    w = w.to(device, non_blocking=True)
                tensors.append(w)
            
            batch_tensor = torch.stack(tensors, dim=0)  # [batch_size, num_experts, o, i] / [batch_size, o, i]

            batch = QuantBatch(
                names=chunk_names,
                tensor=batch_tensor.transpose(-2,-1) if model_type == "qwen3_vl_moe" and "expert" in chunk_names[0] else batch_tensor,
                device=device,
                symmetric=qs.symmetric,
                version=qs.version,
                format=qs.format
            )

            # if model_type is qwen3-vl-moe, the order of input and output (moe) need to be transposed.
            # and the batch.tensor.shape = (batch_size, #experts, input, output)
            if batch.tensor.ndim == 3: # (bs, m, n)
                batch.tensor.unsqueeze_(1)
            
            assert batch.tensor.ndim == 4, f"The len of batch.tensor should be 4, but got {batch.tensor.ndim}"

            if qs.quant_type == "int8":
                temp_qw, temp_s = [], []
                for idx in range(batch.tensor.shape[1]):
                    quant_weight, scale = weight_quant_bf16_to_int8(batch.tensor[:, idx])  # (batch_size, m, n) , (batch_size, m)
                    temp_qw.append(quant_weight.unsqueeze(1))
                    temp_s.append(scale.unsqueeze(1))
                
                quant_weight = torch.cat(temp_qw, dim=1)  # (batch_size, #experts, m, n)
                scale = torch.cat(temp_s, dim=1)  # (bach_size, #experts, m)

                artifacts = QuantBatchArtifacts(
                    names=batch.names,
                    qweight=quant_weight,
                    scale=scale,
                    quant_type=qs.quant_type
                )
            elif qs.quant_type == "int4":
                temp_qw, temp_s, temp_i8scales, temp_i8zeros = [], [], [], []
                i8scales, i8zeros = None, None
                for idx in range(batch.tensor.shape[1]):
                    quant_weight, scale, i8scales, i8zeros = weight_quant_bf16_to_int4pack8(
                        batch.tensor[:, idx],
                        block_size=custom_quant_config.get("block_size", 128),
                        group_size=custom_quant_config.get("group_size", -1),
                        format=qs.format,
                        symmetric=qs.symmetric,
                        version=qs.version
                    )
                    temp_qw.append(quant_weight.unsqueeze(1))
                    temp_s.append(scale.unsqueeze(1))
                    if i8scales is not None:
                        temp_i8scales.append(i8scales.unsqueeze(1))
                        temp_i8zeros.append(i8zeros.unsqueeze(1))
                    
                quant_weight = torch.cat(temp_qw, dim=1)  # (batch_size, #experts, m, n)
                scale = torch.cat(temp_s, dim=1)  # (bach_size, #experts, m, n)

                if len(temp_i8scales) > 0:
                    i8scales = torch.cat(temp_i8scales, dim=1)
                    i8zeros = torch.cat(temp_i8zeros, dim=1)

                artifacts = QuantBatchArtifacts(
                    names=batch.names,
                    qweight=quant_weight,
                    scale=scale,
                    extra={
                        "i8scales": i8scales,
                        "i8zeros": i8zeros,
                    },
                    quant_type=qs.quant_type
                )
            else:
                raise RuntimeError(f"Unsupported quant type {qs.quant_type}")
            
            for name, weight, scale, i8scale, i8zero in quant_post_process(artifacts, qs.version, model_type):
                outputs[name] = weight.cpu()
                outputs[name+"_scale"] = scale.cpu()
                if i8scale is not None:
                    outputs[name+"_i8scales"] = i8scale.cpu()
                if i8zero is not None:
                    outputs[name+"_i8zeros"] = i8zero.cpu()
    
    return outputs, quant_map
