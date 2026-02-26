import re
from .base import QuantConfigBuilder
from ..specs import ResolvedQuantConfig


class DefaultQuantConfigBuilder(QuantConfigBuilder):

    def __init__(self, qconfig: dict, custom_dequant_config: dict):
        self.qconfig = qconfig
        self.groups = qconfig["config_groups"]
        self.custom_ignore = self.get_ignore(custom_dequant_config)
        self.exact_ignore, self.regex_ignore = self.compile_ignore(self.custom_ignore)
    
    def _ignored(self, weight_name: str) -> bool:
        if any(weight_name.startswith(ig) for ig in self.exact_ignore):
            return True
        if any(re.match(p, weight_name) for p in self.regex_ignore):
            return True
        
        return False

    def _match_group(self, weight_name: str):
        for g in self.groups.values():
            for p in g.get("weight_patterns", []):
                if p.startswith("re:") and re.match(p[3:], weight_name):
                    return g
        return None
    
    def build_weight_quant_config(self, qconfig: dict, group_cfg: dict) -> dict:
        wcfg = group_cfg["weights"]

        return {
            "dtype": wcfg["type"],  # int / float
            "num_bits": wcfg["num_bits"],
            "packed": qconfig["format"] == "pack-quantized",
            "strategy": wcfg["strategy"],
            "group_size": wcfg.get("group_size"),
            "block_structure": wcfg.get("block_structure"),
            "symmetric": wcfg["symmetric"],
            "dynamic": wcfg["dynamic"],
        }
    
    def resolve(self, weight_name: str):
        if self._ignored(weight_name):
            return None

        if len(self.groups) == 1:
            group = next(iter(self.groups.values()))
        else:
            group = self._match_group(weight_name)
            if group is None:
                return None

        return ResolvedQuantConfig(
            **self.build_weight_quant_config(self.qconfig, group)
        )
