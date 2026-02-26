import torch
import torch.nn.functional as F

def dequant_fp8_channel_sym(
    artifacts,
    output_dtype=torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    *args,
    **kwargs
) -> torch.Tensor:
    """
    Fully vectorized FP8 channel-wise symmetric dequantization on GPU.
    Channel dimension = out_features
    """
    qweight = artifacts.qweight.to(device, non_blocking=True)
    scale = artifacts.scale.to(device, non_blocking=True)

    # qweight: [out_features, in_features]
    out_features, in_features = qweight.shape

    # -------- scale 处理（channel-wise）--------
    if scale.ndim == 1:
        # [out_features]
        if scale.shape[0] != out_features:
            raise ValueError(
                f"Channel-wise scale must have shape [{out_features}], got {scale.shape}"
            )
        scale_broadcast = scale.view(out_features, 1)

    elif scale.ndim == 2:
        # [out_features, 1]
        if scale.shape == (out_features, 1):
            scale_broadcast = scale
        else:
            raise ValueError(
                f"Unsupported channel-wise scale shape {scale.shape}"
            )
    else:
        raise ValueError(f"Unsupported scale shape {scale.shape}")

    # -------- 向量化反量化 --------
    dequant = qweight.float() * scale_broadcast

    return dequant.to(output_dtype)
