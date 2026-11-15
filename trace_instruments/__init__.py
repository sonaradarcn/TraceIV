"""Approximate online credit-assignment rules for recurrent networks, read as instruments.

The package holds the algorithms and estimators; the scripts that produce the numbers and figures of the
paper live in ``experiments/``, ``analysis/`` and ``plotting/`` at the repository root.

Notation (shared by all modules and by the paper)
-------------------------------------------------
``n``      number of recurrent units                ``d``  input dimension
``q``      fan-in of one unit                       ``P``  number of recurrent parameters (``n * q``)
``B``      number of lanes (independent runs advanced by one batched call)
``theta``  recurrent parameters, shape ``(B, P)``
``J``      exact sensitivity ``d h / d theta``, shape ``(B, n, P)``, propagated by RTRL
``Z``      approximate sensitivity (the trace, i.e. the instrument) kept by a cheaper rule
``delta``  loss derivative with respect to the state, shape ``(B, n)``
``C``      cross-Gram ``E[Z^T F J]``;  ``Sigma = E[Z^T F Z]``;  ``G = E[J^T F J]``
``D``      bias field ``E[(J - Z)^T delta]``
"""

from .config import DTYPE

__version__ = "1.0.0"

__all__ = ["DTYPE", "__version__"]
