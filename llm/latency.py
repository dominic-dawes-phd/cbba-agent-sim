"""Deliberation cost in simulated ticks (hard rule 3).

Tokens are mapped to ticks at fixed reference throughput, never at measured speed, so results do
not depend on the GPU.
"""

import math

from sim.config import LatencyConfig


def latency_ticks(prompt_tokens: int, output_tokens: int, config: LatencyConfig) -> int:
    """Returns the ticks an agent idles while its LLM call is in flight."""
    seconds = prompt_tokens / config.prefill_rate + output_tokens / config.decode_rate
    return math.ceil(seconds / config.seconds_per_tick)
