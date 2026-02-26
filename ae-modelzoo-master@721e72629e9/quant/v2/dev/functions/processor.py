
def quant_post_process(artifacts, version, model_type):
    quant_dtype = artifacts.quant_type
    if artifacts.extra is not None:
        i8scales = artifacts.extra.get("i8scales")
        i8zeros = artifacts.extra.get("i8zeros")
    else:
        i8scales = None
        i8zeros = None

    for idx, name in enumerate(artifacts.names):
        weight = artifacts.qweight[idx]  # [#experts, m, n]
        scale = artifacts.scale[idx]  # [#experts, m, 1]
        i8scale = None if i8scales is None else i8scales[idx]
        i8zero = None if i8zeros is None else i8zeros[idx]

        num_experts = weight.shape[0]
        
        if "expert" in name and "shared" not in name:
            scale = scale.contiguous().view(num_experts, 1, -1) if version == 2 else scale.contiguous()
            if model_type == "qwen3_vl_moe":
                weight = weight.transpose(-2, -1).contiguous()
                if quant_dtype == "int4":
                    scale = scale.transpose(-2, -1).contiguous()
                elif quant_dtype == "int8":
                    scale = scale.transpose(-2, -1).contiguous()
            
            if quant_dtype == "int8":
                scale = scale.transpose(-2, -1).contiguous()

            if i8scale is not None:
                i8scale.squeeze_(0)
                assert i8scale.dim() == 2
            
            if i8zero is not None:
                i8zero.squeeze_(0)
                assert i8zero.dim() == 2
        
        if num_experts == 1:
            weight = weight.squeeze(0)
            scale = scale.squeeze(0)
        
        # TODO: i8scale, i8zero
        
        yield name, weight, scale, i8scale, i8zero