"""Names of the learning rules.

A rule is identified by a string, stored verbatim in the result files:

``rtrl``           real-time recurrent learning: the exact sensitivity ``J``
``snap1``          SnAp-1: ``J`` restricted to the support of the immediate Jacobian
``immediate``      the immediate Jacobian alone (no memory)
``leak:<rate>``    leaky accumulation of the immediate Jacobian, ``Z <- rate * Z + B_t``
``tbptt:<K>``      sensitivity truncated to the last ``K`` steps (truncated backpropagation through time)
``hybrid:<m>``     SnAp-1 for the incoming weights of units ``0..m-1``, exact sensitivity for the others
``uoro``           unbiased online recurrent optimisation: a rank-one unbiased estimate of ``J``

Lanes that add the restoration of the bias field carry the suffix ``+restoration``.
"""

from __future__ import annotations

RTRL = "rtrl"
SNAP1 = "snap1"
IMMEDIATE = "immediate"
UORO = "uoro"

LEAK_PREFIX = "leak:"
TBPTT_PREFIX = "tbptt:"
HYBRID_PREFIX = "hybrid:"


def restored_lane_name(multiple: str) -> str:
    """Name of the SnAp-1 lane restored at ``multiple`` times the threshold tracking rate."""
    return f"snap1+restoration({multiple}x threshold)"


def is_leak(rule: str) -> bool:
    return rule.startswith(LEAK_PREFIX)


def is_tbptt(rule: str) -> bool:
    return rule.startswith(TBPTT_PREFIX)


def is_hybrid(rule: str) -> bool:
    return rule.startswith(HYBRID_PREFIX)


def leak_rate(rule: str) -> float:
    """Leak rate of ``leak:<rate>``."""
    return float(rule.split(":")[1])


def tbptt_window(rule: str) -> int:
    """Window length ``K`` of ``tbptt:<K>``."""
    return int(rule.split(":")[1])


def hybrid_split(rule: str) -> int:
    """Number ``m`` of leading units that keep the SnAp-1 trace in ``hybrid:<m>``."""
    return int(rule.split(":")[1])
