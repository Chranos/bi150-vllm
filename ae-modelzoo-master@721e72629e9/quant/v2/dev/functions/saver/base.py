from abc import ABC, abstractmethod
from typing import Dict
import torch


class BaseSaver(ABC):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    @abstractmethod
    def save(
        self,
        state_dict: Dict[str, torch.Tensor],
        src_file: str,
    ):
        pass
