from .specs import DequantSpec
from core.kernels import (dequant_fp8_block_sym, 
                          dequant_int4_group_sym,
                          dequant_fp8_channel_sym)

DEQUANT_KERNEL_TABLE = {}

DEQUANT_KERNEL_TABLE[
    DequantSpec("float", 8, False, "block", True)
] = dequant_fp8_block_sym


DEQUANT_KERNEL_TABLE[
    DequantSpec("int", 4, True, "group", True)
] = dequant_int4_group_sym


DEQUANT_KERNEL_TABLE[
    DequantSpec("float", 8, False, "channel", True)
] = dequant_fp8_channel_sym