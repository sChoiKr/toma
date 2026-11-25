"""Benchmark configurations for the toy model of manifolds."""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .dgp import (
    DGP,
    Circle, Sphere, Torus, Mobius, SwissRoll,
    Helix, FlatDisk, Concept, Manifold,
)


@dataclass
class BenchmarkConfig:
    """Parameters for a single benchmark scale.

    Attributes
    ----------
    name : str
        Human-readable label (``"small"``, ``"medium"``, …).
    d : int
        Ambient dimension of the superposition space.
    manifolds : list of Manifold
        Ordered list of manifold instances (concepts last).
    n_train : int
        Number of training samples.
    n_test : int
        Number of held-out test samples.
    l0 : int
        Number of active components per sample.
    noise : float
        Gaussian noise std added to ``x``.
    sae_width : int
        Number of SAE dictionary atoms.
    sae_k : int
        TopK sparsity of the SAE.
    """

    name: str
    d: int
    manifolds: List[Manifold]
    n_train: int
    n_test: int
    l0: int
    noise: float
    sae_width: int
    sae_k: int

    @property
    def n_components(self) -> int:
        return len(self.manifolds)

    def build_dgp(self, sigma_bias: float = 0.01) -> DGP:
        """Instantiate and return a :class:`~toma.dgp.DGP` for this config.

        Parameters
        ----------
        sigma_bias : float
            Std of per-component bias offset. Default 0.01.

        Returns
        -------
        DGP
        """
        return DGP(self.manifolds, d=self.d, sigma_bias=sigma_bias)


def _manifold_pool(n_manifolds: int, concept_frac: float = 0.5,
                    seed: int = 0) -> List[Manifold]:
    """Return a list of *n_manifolds* manifolds, half curved-zoo + half Concept.

    Default mix is 50% Concept (1-D line segments, ``ambient_dim=1``) and 50%
    curved manifolds drawn from {Circle, Sphere, Torus, Mobius, SwissRoll,
    Helix, FlatDisk} cycled deterministically.

    Parameters
    ----------
    n_manifolds : int
        Total number of components.
    concept_frac : float, default 0.5
        Fraction of components that should be 1-D Concept atoms.
    seed : int
        NumPy seed used only for choosing the curved-manifold order.
    """
    zoo = [Circle(), Sphere(), Torus(), Mobius(), SwissRoll(), Helix(), FlatDisk()]
    n_concepts = max(1, round(concept_frac * n_manifolds))
    n_zoo = n_manifolds - n_concepts

    rng = np.random.default_rng(seed)
    chosen = [zoo[i % len(zoo)] for i in rng.permutation(n_zoo)]
    concepts = [Concept() for _ in range(n_concepts)]
    return chosen + concepts


# Based on the paper:
# Hardness lever: ``l0 · ⟨k⟩ / d`` controls cross-projection energy at the
# oracle. A linear oracle reaches ~0.95 R² at ratio 0.04, ~0.87 at 0.08,
# ~0.63 at 0.20, ~0.0 at 0.78 (⟨k⟩ ≈ 1.5 with concept_frac=0.7).
#
# All four scales below sit at ratio ≈ 0.09–0.12, with a deliberately
# *aggressive* M/l0 = 15 (lots of competing inactive subspaces). The
# greedy oracle (the realistic ceiling) lands at 0.88–0.90 across the board:
#
#   cfg     d     M    l0   M/l0   greedy R²    greedy F1
#   SMALL   128   150  10   15.0   0.893        0.933
#   MEDIUM  256   300  20   15.0   0.887        0.926
#   LARGE   512   600  40   15.0   0.878        0.918
#   GIANT   1024  900  60   15.0   0.900        0.930

SMALL = BenchmarkConfig(
    name="small",
    d=128,
    manifolds=_manifold_pool(150, concept_frac=0.7),
    n_train=30_000,
    n_test=5_000,
    l0=10,
    noise=0.01,
    sae_width=512,
    sae_k=16,
)

MEDIUM = BenchmarkConfig(
    name="medium",
    d=256,
    manifolds=_manifold_pool(300, concept_frac=0.7),
    n_train=60_000,
    n_test=10_000,
    l0=20,
    noise=0.01,
    sae_width=1_024,
    sae_k=32,
)

LARGE = BenchmarkConfig(
    name="large",
    d=512,
    manifolds=_manifold_pool(600, concept_frac=0.7),
    n_train=120_000,
    n_test=20_000,
    l0=40,
    noise=0.01,
    sae_width=2_048,
    sae_k=64,
)

GIANT = BenchmarkConfig(
    name="giant",
    d=1024,
    manifolds=_manifold_pool(900, concept_frac=0.7),
    n_train=200_000,
    n_test=40_000,
    l0=60,
    noise=0.01,
    sae_width=4_096,
    sae_k=96,
)

CONFIGS = {c.name: c for c in [SMALL, MEDIUM, LARGE, GIANT]}
