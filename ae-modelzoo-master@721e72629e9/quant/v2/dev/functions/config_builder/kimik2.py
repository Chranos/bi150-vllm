# config_builder/qwen3.py
import re

from .base import QuantConfigBuilder
from ..specs import ResolvedQuantConfig


class KimiK2QuantConfigBuilder(QuantConfigBuilder):

    def __init__(self, qconfig: dict, custom_dequant_config: dict):
        qc = qconfig
        self.group_size = qconfig['config_groups']['group_0']['weights']['group_size']
        self.original_ignore = self.get_ignore(qc)
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
            dtype="int",
            num_bits=4,
            packed=True,
            strategy="group",
            group_size=self.group_size,
            block_structure=None,
            symmetric=True,
            dynamic=False,
        )
