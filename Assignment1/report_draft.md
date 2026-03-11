## Classification Trees and Random Forests on TKI Resistance FTIR Spectral Data

**Dataset Overview**
The dataset comprises  Fourier Transform Infrared (FTIR) spectral data used to classify samples into two genetic states: a mutated "Bcr-abl" class (associated with Tyrosine Kinase Inhibitor resistance) and a normal "Wild type" class. It is strongly characterized by high dimensionality, containing 396 continuous spectral features but only 128 training and 60 testing instances. This $p > n$ scenario naturally predisposes standard decision trees to overfit by memorizing noise, making the variance reduction techniques of a random forest especially necessary for robust prediction.

**Part 1: Implementation**
Classification trees are implemented as a `Tree` class that greedily selects splits by minimizing weighted Gini impurity over candidate columns and midpoint thresholds. The tree is stored as nested tuples and traversed recursively for prediction. `RandomForest` builds n trees on bootstrap samples, each using a random subset of √p features per split. A small suite of unit tests covers edge cases such as pure nodes, no valid splits, negative features, single samples, and boundary values of `min_samples`.

**Part 1: Uncertainty Quantification**
Misclassification rate is a Bernoulli proportion, so its standard error is estimated as $SE = \sqrt{\frac{p(1-p)}{n}}$, where p is the observed error rate and n is the test set size. No model rebuilding is needed.

**Part 1: Full Tree Results**

* Train: 0.0000 ± 0.0000, the full tree overfits perfectly to training data
* Test: 0.2667 ± 0.0571, indicating poor generalization

**Part 1: Random Forest Results**

* Train: 0.0000 ± 0.0000
* Test: 0.2333 ± 0.0546, RF improves test error by ~3.3% over the single tree, as expected from variance reduction via bagging and random feature selection

**Part 1: Misclassification vs Number of Trees**
Train error drops quickly to ~0 after ~30 trees. Test error is noisy for small n but stabilizes around 0.22–0.25 after ~40 trees, showing that more trees reduce variance but cannot eliminate bias.

**Part 2: Variable Importance**
Permutation importance is computed per tree on OOB samples. For each variable, its values are permuted and the accuracy drop relative to baseline is recorded, then averaged across all trees. To ensure a valid comparison with root split frequencies, negative accuracy drops (uninformative noise) were clamped to zero, and the final scores were normalized to sum to 1. Variables with high importance are concentrated in spectral regions 260–300 and 340–400. The shuffled root frequency partially overlaps with important variables, which is expected since high-variance spectral regions are both easier to split on by chance and genuinely informative for TKI resistance.
