"""Phase 1.5R2 functional Cox-repair components.

The package keeps the Phase 1.5R runtime implementation intact while making
the mathematical repair and its evidence explicit.  Runtime artifacts are
written below ``outputs/phase1_5r2_functional_repair`` by the runner.
"""

from .cox_gradients import cox_loss_numpy, cox_risk_coefficients
from .rank_gradients import rank_loss_numpy, rank_risk_coefficients

__all__ = [
    "cox_loss_numpy",
    "cox_risk_coefficients",
    "rank_loss_numpy",
    "rank_risk_coefficients",
]
