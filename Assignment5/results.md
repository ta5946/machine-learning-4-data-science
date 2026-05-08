# Results — Kernel Methods

---

## Part 1: Sine Dataset

Parameters were chosen manually. Training MSE measures in-sample fit; support vector counts measure sparsity of the SVR solution.

| Method | Kernel | σ / M | λ | ε | Train MSE | # SVs |
|--------|--------|-------|---|---|-----------|-------|
| KRR | RBF | σ = 1.0 | 0.01 | — | 0.069 | — |
| KRR | Polynomial | M = 5 | 1.0 | — | 0.977 | — |
| SVR | RBF | σ = 1.0 | 0.01 | 0.3 | 0.072 | 85 / 200 |
| SVR | Polynomial | M = 5 | 1.0 | 0.3 | 1.020 | 141 / 200 |

RBF fits the periodic signal well (training MSE 0.07) and produces a sparse solution — most predictions land inside the ε-tube, leaving only 85/200 points as SVs. Polynomial cannot replicate a multi-period sine regardless of parameters, so residuals are large relative to ε and 141/200 points become SVs. The high SV count for polynomial SVR is a direct consequence of kernel mismatch, not a tuning failure.

---

## Part 2: Housing Dataset

**Setup:** 5-fold cross-validation, features standardized (zero mean, unit variance), ε = 2.0 for SVR.  
The *λ from CV* column shows the best MSE found by searching λ ∈ {0.001, 0.01, 0.1, 1, 10, 100} independently for each kernel parameter value.

### KRR — Polynomial kernel

| M | MSE (λ = 1) | MSE (λ from CV) | Best λ |
|---|-------------|-----------------|--------|
| 1 | 41.40 | 41.34 | 0.001 |
| 2 | 28.29 | 27.63 | 0.001 |
| 3 | 45.18 | 41.34 | 10 |
| 4 | 273.27 | 57.95 | 10 |
| 5 | 6648.14 | 142.12 | 100 |
| 6 | 24128.00 | 977.87 | 100 |
| 7 | 1 256 643 | 32 879 | 100 |
| 8 | 5 895 366 | 324 733 | 100 |
| 9 | 4 555 168 | 1 458 166 | 100 |
| 10 | 10 869 021 | 6 559 651 | 100 |

### KRR — RBF kernel

| σ | MSE (λ = 1) | MSE (λ from CV) | Best λ |
|---|-------------|-----------------|--------|
| 0.1 | 507.51 | 491.70 | 0.001 |
| 0.3 | 320.70 | 272.35 | 0.1 |
| 0.5 | 153.98 | 118.34 | 0.1 |
| 1 | 57.26 | 40.40 | 0.01 |
| 2 | 37.18 | 29.95 | 0.01 |
| 5 | 35.67 | 27.93 | 0.01 |
| 10 | 42.20 | 27.13 | 0.001 |

### SVR — Polynomial kernel (ε = 2.0)

| M | MSE (λ = 1) | MSE (λ from CV) | Best λ | # SVs |
|---|-------------|-----------------|--------|-------|
| 1 | 40.84 | 40.08 | 10 | 121 |
| 2 | 26.38 | 25.65 | 10 | 111 |
| 3 | 30.69 | 29.64 | 10 | 108 |
| 4 | 48.42 | 35.07 | 100 | 112 |
| 5 | 637.48 | 41.53 | 100 | 96 |
| 6 | 4047.25 | 207.00 | 100 | 96 |
| 7 | 119 613 | 2840 | 100 | 96 |
| 8 | 1 748 803 | 44 348 | 100 | 85 |
| 9 | 87 437 | 86 995 | 10 | 96 |
| 10 | 691 818 | 691 818 | 0.1 | 75 |

### SVR — RBF kernel (ε = 2.0)

| σ | MSE (λ = 1) | MSE (λ from CV) | Best λ | # SVs |
|---|-------------|-----------------|--------|-------|
| 0.1 | 63.95 | 63.67 | 0.1 | 144 |
| 0.3 | 58.12 | 47.38 | 0.1 | 128 |
| 0.5 | 50.66 | 38.03 | 0.1 | 119 |
| 1 | 38.43 | 28.59 | 0.01 | 98 |
| 2 | 33.18 | 25.64 | 0.001 | 121 |
| 5 | 43.70 | 26.30 | 0.01 | 108 |
| 10 | 407.71 | 26.11 | 0.001 | 114 |

### Best configurations

| Method | Kernel | Best param | Best MSE |
|--------|--------|------------|----------|
| KRR | Polynomial | M = 2 | 27.63 |
| KRR | RBF | σ = 10 | 27.13 |
| SVR | Polynomial | M = 2 | 25.65 |
| SVR | RBF | σ = 2 | 25.64 |

SVR edges out KRR by roughly 2 MSE points across both kernels. Both methods strongly prefer RBF. Polynomial kernels are numerically unstable for M ≥ 4 on this (un-normalized) feature scale, and even λ = 100 cannot recover good performance beyond M ≈ 6.

---

## Comparison: KRR vs SVR

**Similarities.** Both methods use the same kernel trick and produce identical predictions in the limit of small λ and ε → 0. They respond to regularization in the same direction — larger λ means smoother, more biased fits — and both benefit substantially from CV-tuned λ over the fixed λ = 1 baseline.

**Differences.**

*Fit quality.* SVR consistently achieves lower MSE (~25.6 vs ~27.1 at best), a modest but consistent improvement. The ε-insensitive loss ignores small residuals, making SVR more robust to noise near the fit.

*Sparsity.* SVR produces a sparse solution — only points outside the ε-tube become support vectors and contribute to the prediction. With ε = 2.0, between 75 and 144 of 300 points are SVs depending on the kernel and its parameter. KRR uses all training points with no notion of sparsity.

*Computational cost.* KRR requires solving one linear system per fit (O(n³) once). SVR requires solving a quadratic program, which is slower and numerically more demanding — for n = 300 it is still tractable, but it would become a bottleneck at larger scales.

*Parameter sensitivity.* KRR has two parameters (kernel param + λ); SVR adds ε. Choosing ε requires domain knowledge about the acceptable residual scale. If ε is poorly set (too small → dense, too large → underfit), SVR degrades more sharply than KRR.

**Preference.** For this dataset, KRR is the more practical choice. The MSE gap is small (≈2 points), KRR is faster to fit and has one fewer hyperparameter to tune. SVR's sparsity advantage is valuable when prediction-time cost matters (fewer support vectors = faster `predict`) or when the dataset is noisy and robustness to outliers is important. At n = 300, neither concern is pressing, so the simpler method wins.
