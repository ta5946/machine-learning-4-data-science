# AGENTS.md - Kernel Methods Assignment

## Course Context

**Course:** Machine Learning for Data Science 1 (MLDS1), 2025-2026
**Institution:** University of Ljubljana, Faculty of Computer and Information Science (FRI)
**Assignment:** Homework 5 - Kernel Regression Methods
**Author:** Tjaš Ajdovec (ta5946@fri.uni-lj.si)

This is a graduate-level machine learning course covering supervised learning methods. Previous assignments covered topics like neural networks (HW4). This assignment focuses on kernel methods for regression.

## Directory Structure

```
Assignment5/
├── AGENTS.md                 # This file - project documentation for AI agents
├── instructions.md           # PRIMARY: Assignment requirements and grading criteria
├── test_kernels.py           # Unit tests defining required code interface
├── hw_kernels.py             # TO CREATE: Main implementation file
│
├── sine.csv                  # Dataset for Part 1 (1D regression)
├── housing2r.csv             # Dataset for Part 2 (multi-dim regression)
│
├── 060-kernel_notes.pdf      # Lecture notes on kernel methods
├── 062-quadratic_programming_notes.pdf  # QP optimization theory
├── SmoSch04.pdf              # Smola & Scholkopf 2004 - SVR reference (USE Eq. 10)
├── 2003-gartner-a-survey-of-kernels-for-structure-data.pdf  # Structured kernels (Part 3)
│
├── last_homework/            # Reference for code and report style
│   ├── nn.py                 # NumPy neural network - CODE STYLE REFERENCE
│   ├── nn_pt.py              # PyTorch neural network implementation
│   ├── compare_nn.py         # Comparison script between implementations
│   └── hw4.tex               # LaTeX report - REPORT STYLE REFERENCE
│
└── lecture_code/             # Instructor-provided lecture examples
```

## File Descriptions

### Core Assignment Files

| File | Purpose | Status |
|------|---------|--------|
| `instructions.md` | Complete assignment specification with all requirements | Read first |
| `test_kernels.py` | Unit tests that define the required API | Must pass all tests |
| `hw_kernels.py` | Your implementation | To be created |

### Datasets

| File | Dimensions | Features | Target | Used In |
|------|------------|----------|--------|---------|
| `sine.csv` | ~100 rows | `x` (1D) | `y` | Part 1 - visualization |
| `housing2r.csv` | ~300 rows | `RM`, `AGE`, `DIS`, `RAD`, `TAX` | `y` | Part 2 - evaluation |

### Reference PDFs

| File | Content | Relevance |
|------|---------|-----------|
| `060-kernel_notes.pdf` | Kernel methods theory | Background reading |
| `062-quadratic_programming_notes.pdf` | QP optimization | SVR implementation |
| `SmoSch04.pdf` | SVR tutorial paper | **Critical: Use Eq. (10) for SVR** |
| `2003-gartner-a-survey-of-kernels-for-structure-data.pdf` | Kernels for graphs, text, etc. | Part 3 inspiration |

### Reference Code (last_homework/)

| File | Purpose | What to Learn |
|------|---------|---------------|
| `nn.py` | NumPy ANN implementation | Code style, class structure, documentation |
| `nn_pt.py` | PyTorch ANN | Alternative implementation patterns |
| `compare_nn.py` | Comparison utilities | Visualization, timing, evaluation |
| `hw4.tex` | LaTeX report | Report structure, formatting, conciseness |

---

## Assignment Requirements

### Part 1 (Required - Base Grade)

Implement two regression methods with two kernels:

**Regression Methods:**
1. **Kernelized Ridge Regression** - Standard kernel ridge regression
2. **Support Vector Regression (SVR)** - Using quadratic programming via `cvxopt.solvers.qp`

**Kernels:**
1. **Polynomial kernel:** `κ(x, x') = (1 + x·x')^M`
2. **RBF kernel:** `κ(x, x') = exp(-||x - x'||² / 2σ²)`

**SVR Implementation Details:**
- Solve optimization from Eq. (10) in Smola & Scholkopf 2004 (`SmoSch04.pdf`)
- QP solution vector ordering: `[α₁, α₁*, α₂, α₂*, α₃, α₃*, ...]`
- Set `C = 1/λ`
- Obtain `b` from cvxopt output `y`, NOT from Eq. (16)

**Part 1 Deliverable:**
- Apply both methods and kernels to `sine.csv` dataset
- Find good kernel and regularization parameters manually (no cross-validation required)
- Plot: input data, fit curve, marked support vectors
- Aim for sparse SVR solutions

### Part 2 (Grades 7-8)

Apply methods to `housing2r.csv` dataset:
- Measure MSE performance
- Plot MSE vs kernel parameter:
  - Polynomial: M ∈ [1, 10]
  - RBF: choose interesting σ values
- Two curves per plot: λ=1 and λ from internal cross-validation
- Display number of support vectors for SVR
- Compare ridge regression vs SVR with analysis

### Part 3 (Grades 9-10)

Structured data regression:
- Find/create regression dataset with structured data (text, images, graphs, sound)
- Implement a kernel that cannot be trivially replicated with tabular attributes
- Compare kernel approach to naive attribute representation
- Submit as separate `.zip` with code and data

---

## Required Code Interface

The code must be in a single file `hw_kernels.py` conforming to `test_kernels.py`:

```python
# Kernel classes - must support vectors (1D) and matrices (2D) without Python loops
class Polynomial:
    def __init__(self, M):
        self.M = M  # Polynomial degree

    def __call__(self, A, B):
        # Returns: scalar (1D,1D), 1D array (1D,2D or 2D,1D), 2D array (2D,2D)
        pass

class RBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, A, B):
        # Hint for vectorized distance: (a-b)² = a·a - 2a·b + b·b
        pass

# Regression classes
class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_):
        pass

    def fit(self, X, y):
        # Returns a model object with predict(X) method
        pass

class SVR:
    def __init__(self, kernel, lambda_, epsilon):
        pass

    def fit(self, X, y):
        # Returns a model with predict(X), get_alpha(), get_b() methods
        # get_alpha() returns shape (n_samples, 2) - [αᵢ, αᵢ*] per sample
        pass
```

---

## Code Style Reference

Based on `last_homework/nn.py`:
- Clear function/class documentation with comments explaining purpose
- NumPy-based vectorized operations (no Python loops in kernels)
- Separate fitter classes and model classes
- Helper functions for data I/O and utilities
- Descriptive variable names

Example pattern:
```python
class FitterClass:
    def __init__(self, params):
        self.params = params

    def fit(self, X, y):
        # Training logic
        return ModelClass(learned_params)

class ModelClass:
    def __init__(self, params):
        self._params = params

    def predict(self, X):
        # Prediction logic
        return predictions
```

---

## Report Style Reference

Based on `last_homework/hw4.tex`:
- LaTeX with FRI DS report template
- **Strict 1-page limit per part**
- Structure: Introduction, sections per part, results tables/figures
- Include: equations, figures with captions, tables for numerical results
- Concise technical writing
- Reference bibliography where appropriate

---

## Dependencies

**Allowed for core implementation:**
- Python 3.12
- `numpy`
- `cvxopt` (for QP solving in SVR)

**Allowed for auxiliary tasks:**
- Any library for data loading, visualization, cross-validation, scoring
- Common choices: `pandas`, `matplotlib`, `scikit-learn` (for CV/metrics only)

---

## Submission Checklist

- [ ] `hw_kernels.py` - Single Python file with all implementations
- [ ] Report PDF (max 3 pages total: 1 per part)
- [ ] Part 3 ZIP file with structured data code and dataset
- [ ] All tests in `test_kernels.py` pass

## Running Tests

```bash
python test_kernels.py
```

---

## Key Implementation Notes

1. **Kernels must be vectorized** - No Python for-loops, use NumPy broadcasting
2. **RBF distance trick:** `||a-b||² = a·a - 2a·b + b·b` for vectorization
3. **SVR alpha ordering is critical** - Tests verify `[α₁, α₁*, α₂, α₂*, ...]`
4. **SVR b from cvxopt** - Use dual variable `y` from QP solution, not Eq. 16
5. **Epsilon handling** - Set ε properly for good SVR fit and sparsity
6. **Cross-validation** - Use internal CV for λ selection in Part 2

---

## Mathematical Background

### Kernelized Ridge Regression
Given kernel matrix K where K_ij = κ(x_i, x_j):
- Solve: `α = (K + λI)^(-1) y`
- Predict: `f(x) = Σ α_i κ(x_i, x)`

### Support Vector Regression
Dual formulation from Smola & Scholkopf Eq. (10):
- Minimize quadratic objective with box constraints
- Support vectors: points where α_i or α_i* > 0
- Prediction: `f(x) = Σ (α_i - α_i*) κ(x_i, x) + b`

### Kernel Properties
- Polynomial: captures feature interactions up to degree M
- RBF: infinite-dimensional feature space, locality controlled by σ
- Both must satisfy Mercer's condition (positive semi-definite)
