"""TriPrune-HGNN: adaptive hypergraph pruning with learned threshold control
and attention-based negative mining.

Reference implementation accompanying the TMLR paper. See README.md for scope.
"""
from .data.hetero_hypergraph import HeteroHypergraph
from .data.synthetic import generate_synthetic_hypergraph, empirical_compressibility
from .models.hcp import HCP, train_hcp
from .models.triprune import TriPruneHGNN
from .training.curriculum import TrainConfig, train_triprune, evaluate

__version__ = "0.1.0"
__all__ = [
    "HeteroHypergraph", "generate_synthetic_hypergraph", "empirical_compressibility",
    "HCP", "train_hcp", "TriPruneHGNN", "TrainConfig", "train_triprune", "evaluate",
]
