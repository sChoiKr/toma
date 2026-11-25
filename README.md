# Toy model of Manifolds (TOMA)

This is a minimal benchmark for studying how block-sparse featurizers recover low-dimensional manifold structure from superposed representations.

**Data generating process**

```
x = Σᵢ mᵢ + ε
```

Each `mᵢ` is a manifold contribution: a point sampled from a low-dimensional shape (circle, sphere, torus, …) and embedded in ℝᵈ via a random orthonormal map. ~70 % of components are `Concept` atoms (1-D line segments, classic dictionary learning); the rest are drawn from the curved manifold zoo. Only `l0` components are active per sample.

- **`toma`** — synthetic benchmark: manifold zoo, DGP, four pre-defined scales (SMALL → GIANT), and ground-truth recovery metrics.
- **`toma.bsf`** — block-sparse featurizers: `VanillaBSF` (free encoder + block TopK) and `GrassmannianBSF` (tied orthonormal frames + block TopK). One trainer works for both.

> **Start here:** the notebooks in [`runs/small/`](runs/small/) — `01_block`, `02_grassmannian`, `03_sae` — end-to-end walkthroughs: manifold zoo viz, training, recovery metrics, and a 3-D recovery figure. `03_sae` trains a classical SAE (one atom per block) as a baseline for the two block-sparse featurizers.

---

## install

```bash
git clone https://github.com/sChoiKr/toma
cd toma
pip install -e .          # installs the `toma` package (and deps)
```

Now `import toma` works from anywhere. `bsf` is the `toma.bsf` submodule, so the
notebooks under `runs/` just do `import toma` / `from toma import bsf`.

---

## quick start

```python
import numpy as np
from toma.configs import SMALL

np.random.seed(42)
dgp = SMALL.build_dgp()

np.random.seed(0)
data = dgp.sample(n=SMALL.n_train, l0=SMALL.l0, noise=SMALL.noise)

x        = data["x"]             # (N, d)
contribs = data["contributions"] # (N, M, d)  — mᵢ per sample
masks    = data["active_masks"]  # (N, M) bool
thetas   = data["thetas"]        # list[M], intrinsic coords per component
```

---

## manifolds

| class | ambient dim | intrinsic dim |
|-------|-------------|---------------|
| `Circle` | 2 | 1 |
| `Sphere` | 3 | 2 |
| `Hypersphere(n)` | n+1 | n |
| `Torus` | 3 | 2 |
| `Mobius` | 3 | 2 |
| `SwissRoll` | 3 | 2 |
| `Helix` | 3 | 1 |
| `FlatDisk` | 2 | 2 |
| `LineSegment` / `Concept` | 1 | 1 |

---

## benchmark configs

~70 % of components are `Concept` atoms; the rest are curved manifolds from the zoo. All four scales sit at cross-projection ratio ≈ 0.09–0.12 with M/l0 = 15. The greedy oracle lands at R² ≈ 0.88–0.90, F1 ≈ 0.92–0.93 across scales.

| config | d | M (components) | l0 | n_train | n_test |
|--------|---|----------------|----|---------|--------|
| `SMALL` | 128 | 150 | 10 | 30k | 5k |
| `MEDIUM` | 256 | 300 | 20 | 60k | 10k |
| `LARGE` | 512 | 600 | 40 | 120k | 20k |
| `GIANT` | 1024 | 900 | 60 | 200k | 40k |

```python
from toma.configs import MEDIUM
import numpy as np

np.random.seed(42); dgp  = MEDIUM.build_dgp()
np.random.seed(0);  data = dgp.sample(n=MEDIUM.n_train, l0=MEDIUM.l0, noise=MEDIUM.noise)
```

---

## block-sparse featurizers

A *concept* is a block of `group_size` latent dims. Both featurizers share a linear decoder and a single featurizer-agnostic trainer:

| featurizer | idea |
|---|---|
| `VanillaBSF` | free encoder, per-sample block TopK |
| `GrassmannianBSF` | tied orthonormal frame per concept + block TopK + learned scale `gamma` |

With `group_size=2`, each concept is a 2-D subspace — a natural match for `Concept` atoms (1-D) and most curved manifolds (intrinsic dim 1–2). Both featurizers use `n_groups = 2M` (2× overcomplete dictionary).

```python
import numpy as np, torch
from toma import bsf
from toma.configs import SMALL

cfg = SMALL
np.random.seed(42); dgp   = cfg.build_dgp()
np.random.seed(0);  train = dgp.sample(n=cfg.n_train, l0=cfg.l0, noise=cfg.noise)
np.random.seed(1);  test  = dgp.sample(n=cfg.n_test,  l0=cfg.l0, noise=cfg.noise)

scale = float(np.sqrt((train['x'] ** 2).sum(-1).mean()))
x_tr  = train['x'] / scale
x_te  = test['x']  / scale

model = bsf.GrassmannianBSF(d=cfg.d, n_groups=2 * cfg.n_components, group_size=2, l0=cfg.l0)
bsf.train(model, x_tr, epochs=300)

with torch.no_grad():
    z = model.encode(torch.as_tensor(x_te, dtype=torch.float32))
codes      = z.reshape(z.shape[0], -1).cpu().numpy()          # (N, G*K)
dictionary = model.decoder_atoms().detach().cpu().numpy()      # (G*K, d)
```

---

## recovery metrics

```python
from toma.metrics import best_group_per_component, group_oracle_r2
import numpy as np

m_te  = test['contributions'] / scale   # (N, M, d)
masks = test['active_masks']             # (N, M) bool

best_g, f1 = best_group_per_component(codes, model.n_groups, model.group_size, masks, dgp.components)
r2 = group_oracle_r2(codes, dictionary, model.group_size, m_te, masks, dgp.components, best_groups=best_g)

print('group-oracle R²', np.nanmean(r2))
print('group F1       ', f1.mean())
```

**group-oracle R²** — for each concept *i*, find the group whose on/off firing best predicts when *i* is active (by F1), reconstruct `mᵢ` from only that group's atoms, and compute R² against the true contribution.

**group F1** — for each concept *i*, find the group whose block norm best predicts `active_masks[:, i]` and report the F1 of that predictor.

Also available for atom-level (non-block) scoring: `oracle_k_r2` and `component_f1`.

---

## layout

```
toma/
  dgp.py        manifold zoo + sparse mixture DGP
  configs.py    SMALL / MEDIUM / LARGE / GIANT benchmark scales
  metrics.py    recovery metrics (atom- and block-level R² + F1)
  bsf/          block-sparse featurizers (the toma.bsf submodule)
    base.py          abstract BSF (decode / forward / atoms / block ops)
    vanilla.py       VanillaBSF   — free encoder + block TopK
    grassmannian.py  GrassmannianBSF — orthonormal frames + block TopK
    train.py         featurizer-agnostic trainer + recon_r2
runs/small/
  01_block.ipynb          end-to-end: VanillaBSF
  02_grassmannian.ipynb   end-to-end: GrassmannianBSF
  03_sae.ipynb            end-to-end: classical SAE baseline (ReLU + TopK, group_size=1)
```

Everything runs on CPU; a GPU just makes training faster.

## requirements

`torch`, `numpy`, `matplotlib` (`nbformat` only to regenerate the notebooks).
