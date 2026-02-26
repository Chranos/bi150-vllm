import torch
import time
import numpy as np

from pathlib import Path

current_dir = Path(__file__).resolve()
obj_dir = current_dir.parent.parent

import sys
sys.path.insert(0, str(obj_dir))

from pytorchs.fp8_block_sys import dequant_fp8_block_sym
from tritons.fp8_block_sys import dequant_fp8_block_sym_triton

# 假设你已经有以下函数
# dequant_fp8_block_sym    : PyTorch 实现
# dequant_fp8_block_sym_triton : Triton 实现

# -------------------------------
# 测试参数
M, N = 1024, 1024       # 权重矩阵大小
block_size = 32

# 随机生成量化权重
torch.manual_seed(0)
qweight = torch.randint(-127, 128, (M, N), dtype=torch.int8, device='cuda')

# 随机生成 block-wise scale
num_blocks = (N + block_size - 1) // block_size
scale = torch.rand((M, num_blocks), dtype=torch.float32, device='cuda')

# 构造 artifacts 和 RQ
from dataclasses import dataclass

@dataclass
class QuantArtifacts:
    qweight: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor | None = None
    shape: torch.Tensor | None = None

@dataclass
class ResolvedQuantConfig:
    dtype: str
    num_bits: int
    packed: bool
    strategy: str
    group_size: int | None
    block_structure: tuple | None
    symmetric: bool
    dynamic: bool

artifacts = QuantArtifacts(qweight=qweight, scale=scale)
rq = ResolvedQuantConfig(
    dtype="float",
    num_bits=8,
    packed=False,
    strategy="block",
    group_size=None,
    block_structure=(M, block_size),
    symmetric=True,
    dynamic=False
)

# -------------------------------
# PyTorch 反量化
torch.cuda.synchronize()
t0 = time.time()
dequant_pt = dequant_fp8_block_sym(artifacts, rq)
torch.cuda.synchronize()
t_pt = time.time() - t0
print(f"PyTorch dequant time: {t_pt*1000:.2f} ms")
# -------------------------------
# Triton 反量化
torch.cuda.synchronize()
t0 = time.time()
dequant_tr = dequant_fp8_block_sym_triton(artifacts, rq)
torch.cuda.synchronize()
t_tr = time.time() - t0
print(f"Triton dequant time: {t_tr*1000:.2f} ms")

# -------------------------------
# 验证数值一致性
diff = (dequant_pt.float() - dequant_tr.float()).abs().max().item()
print(f"Max absolute difference: {diff}")
if diff < 1e-5:
    print("✅ PyTorch and Triton results match!")
else:
    print("❌ Results mismatch!")

# -------------------------------
# 可选：统计平均误差
mean_diff = (dequant_pt.float() - dequant_tr.float()).abs().mean().item()
print(f"Mean absolute difference: {mean_diff}")
