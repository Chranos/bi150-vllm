# config_builder/qwen3.py
import re

from .base import QuantConfigBuilder
from ..specs import ResolvedQuantConfig


class Qwen3MoEQuantConfigBuilder(QuantConfigBuilder):

    def __init__(self, qconfig: dict, custom_dequant_config: dict):
        qc = qconfig
        self.block_h, self.block_w = qc["weight_block_size"]
        # get ignore from qconfig
        self.original_ignore = self.get_ignore(qc)
        # get ignore from custom_dequant_config
        self.custom_ignore = self.get_ignore(custom_dequant_config)
        self.exact_ignore, self.regex_ignore = self.compile_ignore(self.original_ignore+self.custom_ignore)

    def _ignored(self, weight_name: str) -> bool:
        if any(weight_name.startswith(ig) for ig in self.exact_ignore):
            return True
        if any(re.match(p, weight_name) for p in self.regex_ignore):
            return True
        
        return False

    def resolve(self, weight_name: str):
        if self._ignored(weight_name):
            return None

        return ResolvedQuantConfig(
            dtype="float",
            num_bits=8,
            packed=False,
            strategy="block",
            group_size=None,
            block_structure=(self.block_h, self.block_w),
            symmetric=True,
            dynamic=False,
        )
