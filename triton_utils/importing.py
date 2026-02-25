# SPDX-License-Identifier: Apache-2.0

from importlib.util import find_spec

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

HAS_TRITON = False
