import re
import torch


def compile_ignore(ignore_list: list | None = None, 
                   extra_ignore_list: list | None = None):
    exact = set()
    regex = []

    for x in (ignore_list or []) + (extra_ignore_list or []):
        if x.startswith("re:"):
            regex.append(re.compile(x[3:]))
        else:
            exact.add(x)
    return exact, regex


def should_ignore(weight_name, exact_ignore: list, regex_ignore: list):
    if any(weight_name.startswith(ig) for ig in exact_ignore):
        return True
    if any(re.match(p, weight_name) for p in regex_ignore):
        return True
    
    return False


def should_quant(weight_name, exact_pattern: list, regex_pattern: list):
    return should_ignore(weight_name, exact_pattern, regex_pattern)

    
def get_ignore(quantization_config: dict):
    ignore = []

    candidate_suffixes = ["modules_to_not_convert", "ignore"]
    for cs in candidate_suffixes:
        ig = quantization_config.get(cs)
        if ig:
            ignore.extend(ig)

    return ignore


def get_format(weight_name, format):
    tn_group = format['TN']
    if tn_group:
        exact, regex = compile_ignore(tn_group)
        if should_ignore(weight_name, exact, regex):
            return "TN"
    return "NN"


def estimate_gpu_bytes(
    artifacts,
    rq,
    output_dtype=torch.bfloat16,
) -> int:
    """
    Conservative estimation of peak GPU memory usage (bytes)
    during dequantization of ONE weight.

    Used for batching / flushing decision only.
    """

    bytes_total = 0

    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    qweight = artifacts.qweight
    scale = artifacts.scale

    bytes_total += tensor_bytes(qweight)
    bytes_total += tensor_bytes(scale)

    if artifacts.zero_point is not None:
        bytes_total += tensor_bytes(artifacts.zero_point)

    bytes_total += qweight.numel() * 4  # fp32

    out_elem_size = torch.tensor([], dtype=output_dtype).element_size()
    bytes_total += qweight.numel() * out_elem_size

    SAFETY_FACTOR = 1.3
    return int(bytes_total * SAFETY_FACTOR)