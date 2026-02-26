import torch

from dataclasses import dataclass


@dataclass(frozen=True)
class DequantSpec:
    dtype: str  # "int" | "float"
    num_bits: int  # 4 / 8
    packed: bool  # True / False
    strategy: str  # "group" | "block" | "channel" | "tensor"
    symmetric: bool  # True / False


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


@dataclass
class QuantArtifacts:
    qweight: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor | None
    shape: torch.Tensor | None


@dataclass(frozen=True)
class QuantSpec:
    quant_type: str
    shape: tuple
    symmetric: bool = True
    version: int = 2
    format: str = "TN"


@dataclass
class QuantBatch:
    names: list[str]
    tensor: torch.Tensor
    device: torch.device
    symmetric: bool = True
    version: int = 2
    format: str = "TN"


@dataclass
class QuantBatchArtifacts:
    names: list[str]
    qweight: torch.Tensor
    scale: torch.Tensor
    quant_type: str | None = None
    extra: dict | None = None