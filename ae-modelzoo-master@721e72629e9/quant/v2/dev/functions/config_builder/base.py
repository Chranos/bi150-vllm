# config_builder/base.py
import re

from typing import Optional
from ..specs import ResolvedQuantConfig


class QuantConfigBuilder:

    def resolve(
        self,
        weight_name: str,
    ) -> Optional[ResolvedQuantConfig]:
        pass
    
    def get_ignore(self, quantization_config: dict):
        ignore = []

        candidate_suffixes = ["modules_to_not_convert", "ignore"]
        for cs in candidate_suffixes:
            ig = quantization_config.get(cs)
            if ig:
                ignore.extend(ig)

        return ignore
    
    def compile_ignore(self, ignore_list):
        exact = set()
        regex = []

        for x in (ignore_list or []):
            if x.startswith("re:"):
                regex.append(re.compile(x[3:]))
            else:
                exact.add(x)
        return list(exact), regex