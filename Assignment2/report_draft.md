## Before deduplication:

| Model                    | Tuning Approach             | Classification Accuracy | Log Loss        |
|:-------------------------|:----------------------------|:------------------------|:----------------|
| **DummyClassifier**      | Relative frequency baseline | 0.6081 ± 0.0141         | 1.1658 ± 0.0326 |
| **LogisticRegression**   | Train fold optimization `C` | 0.7357 ± 0.0159         | 0.6711 ± 0.0383 |
| **LogisticRegression**   | Nested cross-validation `C` | 0.7369 ± 0.0156         | 0.6715 ± 0.0387 |
| **KNeighborsClassifier** | Train fold optimization `k` | 0.6933 ± 0.0185         | 6.1686 ± 0.6617 |
| **KNeighborsClassifier** | Nested cross-validation `k` | 0.6986 ± 0.0223         | 0.9698 ± 0.0906 |

## After deduplication with 5 folds:
- Lower accuracy (no seen instances)

| Model                    | Tuning Approach             | Classification Accuracy | Log Loss        |
|:-------------------------|:----------------------------|:------------------------|:----------------|
| **DummyClassifier**      | Relative frequency baseline | 0.6090 ± 0.0154         | 1.1698 ± 0.0321 |
| **LogisticRegression**   | Train fold optimization `C` | 0.7312 ± 0.0159         | 0.6798 ± 0.0346 |
| **LogisticRegression**   | Nested cross-validation `C` | 0.7308 ± 0.0158         | 0.6795 ± 0.0347 |
| **KNeighborsClassifier** | Train fold optimization `k` | 0.6833 ± 0.0147         | 6.3590 ± 0.4812 |
| **KNeighborsClassifier** | Nested cross-validation `k` | 0.6951 ± 0.0180         | 0.9999 ± 0.0926 |

## With 10 outer folds:
- Higher accuracy (more training data)
- Higher variance (fewer instances in test folds)

| Model                    | Tuning Approach             | Classification Accuracy | Log Loss        |
|:-------------------------|:----------------------------|:------------------------|:----------------|
| **DummyClassifier**      | Relative frequency baseline | 0.6090 ± 0.0221         | 1.1697 ± 0.0385 |
| **LogisticRegression**   | Train fold optimization `C` | 0.7314 ± 0.0211         | 0.6794 ± 0.0420 |
| **LogisticRegression**   | Nested cross-validation `C` | 0.7310 ± 0.0222         | 0.6802 ± 0.0419 |
| **KNeighborsClassifier** | Train fold optimization `k` | 0.6856 ± 0.0181         | 6.3011 ± 0.4885 |
| **KNeighborsClassifier** | Nested cross-validation `k` | 0.6966 ± 0.0210         | 0.9737 ± 0.1171 |

## Competition distribution:
- NBA is underrepresented in our sample

| Competition | Dataset Frequency | Real World Frequency | Difference |
|:------------|:------------------|:---------------------|:-----------|
| **NBA**     | 0.2482            | 0.6000               | -0.3518    |
| **EURO**    | 0.2707            | 0.1000               | 0.1707     |
| **SLO1**    | 0.2318            | 0.1000               | 0.1318     |
| **U14**     | 0.1474            | 0.1000               | 0.0474     |
| **U16**     | 0.1019            | 0.1000               | 0.0019     |

## Accuracy by competition:
- NBA is the easiest to predict

| Model                    | EURO  | NBA   | SLO1  | U14   | U16   |
|:-------------------------|:------|:------|:------|:------|:------|
| **DummyClassifier**      | 0.622 | 0.674 | 0.627 | 0.513 | 0.515 |
| **LogisticRegression**   | 0.740 | 0.774 | 0.681 | 0.713 | 0.739 |
| **KNeighborsClassifier** | 0.706 | 0.732 | 0.679 | 0.649 | 0.680 |

## Estimated real world performance:

| Model                    | Dataset Accuracy | Estimated Real World | Change  |
|:-------------------------|:-----------------|:---------------------|:--------|
| **DummyClassifier**      | 0.6090           | 0.6320               | +0.0230 |
| **LogisticRegression**   | 0.7308           | 0.7519               | +0.0211 |
| **KNeighborsClassifier** | 0.6951           | 0.7104               | +0.0153 |


## Discussion on Spatial Cross-Validation Strategies
**Reference:** Mahoney, M. J., et al. (2023). *Assessing the performance of spatial cross-validation approaches for models of spatially structured data.*

#### 1. The Challenge of Model Evaluation on Spatial Data
The fundamental challenge of evaluating spatial models lies in spatial autocorrelation—the statistical principle that near things are more related than distant things. 

* **Contrast with IID Data:** Standard machine learning models assume data points are Independent and Identically Distributed. If spatial data is randomly split, the analysis and assessment sets remain physically entangled. The training data essentially "whispers" the answers to the proximal test data. Consequently, the model merely interpolates between known nearby points rather than learning generalizable environmental rules, resulting in a severe optimistic bias.
* **Contrast with Temporal Data:** Temporal data also exhibits autocorrelation, but time is 1-dimensional and strictly directional. To prevent information leakage in time-series forecasting, researchers can establish a clean cutoff date and test exclusively on future events. Space, however, is multidimensional and omnidirectional. An assessment point is surrounded by training data from every direction, meaning a simple planar cutoff fails. Researchers must explicitly carve out complex exclusion moats to prevent multidirectional information leakage.

#### 2. Spatial Cross-Validation Methods: Mechanics and Trade-offs
The methods evaluated in the paper attempt to enforce spatial independence by physically separating the assessment set ($D_{out}$) from the analysis set ($D_{in}$).

| Method                                        | Technical Mechanics                                                                                                                                                | Advantages                                                                                        | Disadvantages                                                                                                                  |
|:----------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------|
| **Resubstitution**                            | $D_{in}$ and $D_{out}$ are identical. The model is evaluated on the exact dataset used for training.                                                               | Maximizes sample size. Mathematically simplest to execute.                                        | Zero spatial independence. Tests spatial memorization rather than predictive extrapolation.                                    |
| **Ordinary V-fold**                           | Partitions data randomly into $V$ folds, completely ignoring spatial coordinates.                                                                                  | Standardized implementation across all standard ML frameworks.                                    | Fails to break spatial autocorrelation; distances between $D_{in}$ and $D_{out}$ are routinely near zero.                      |
| **Spatial Block CV**                          | Partitions the continuous study area using a regular grid geometry and assigns entire blocks to assessment folds.                                                  | Straightforward implementation and easy visual communication.                                     | Grid boundaries artificially dissect continuous features. Highly sensitive to the block size and spatial alignment parameters. |
| **Spatial Clustered CV**                      | Applies unsupervised clustering algorithms (e.g., k-means) to spatial coordinates to group proximal data into folds.                                               | Folds respect the empirical spatial density and natural geometry of the data.                     | Can yield highly imbalanced fold sizes in skewed or clumped sampling designs.                                                  |
| **Buffered Leave-One-Observation-Out (BLO3)** | Iteratively sets $D_{out}$ to a single observation. Excludes all training points from $D_{in}$ that fall within a fixed spatial buffer radius around it.           | Retains maximum permissible data in the analysis set for training.                                | Computationally expensive. Single-point assessment provides high-variance error estimates.                                     |
| **Leave-One-Disc-Out (LODO)**                 | Iteratively sets $D_{out}$ using an inclusion radius (a spatial disc of points). $D_{in}$ is restricted by an explicit exclusion buffer parameter around the disc. | Realistically simulates the geographic isolation of predicting a completely novel spatial extent. | Extreme sensitivity to the manually specified exclusion buffer radius.                                                         |

#### 3. Empirical Performance of CV Strategies
The paper demonstrates that methods failing to account for spatial distance severely underestimate error, while Clustered CV provides the most robust estimation. 

Below are the empirical results from the simulation, comparing the estimated RMSE against the **Ideal RMSE of 0.715 (0.042)**. The target metric denotes how frequently each method's estimate fell within the true error range.

| Method             | RMSE Estimate | % Within Target |
|:-------------------|:--------------|:----------------|
| **Clustered**      | 0.743 (0.161) | 90.00%          |
| **LODO**           | 0.641 (0.135) | 36.97%          |
| **Blocked**        | 0.664 (0.159) | 31.70%          |
| **BLO3CV**         | 0.440 (0.076) | 27.90%          |
| **V-fold**         | 0.429 (0.098) | 2.00%           |
| **Resubstitution** | 0.189 (0.032) | 0.00%           |

**Summary of Results:**
* **Primary Winner:** Clustered CV was the clear winner, successfully estimating the target error rate 90% of the time without requiring complex parameter tuning.
* **Conditional Performers:** LODO and Blocked CV were moderately successful but highly dependent on perfect parameter tuning to avoid pessimistic bias (over-exclusion) or optimistic bias (under-exclusion).
* **Failure Cases:** Resubstitution and V-fold completely failed to estimate the true extrapolation error, suffering from massive optimistic bias.

#### 4. Advanced Approaches: Nearest Neighbor Distance Matching (NNDM)
The most prominent modern approach missing from this paper is **Nearest Neighbor Distance Matching (NNDM)**. 

While the spatial methods in the paper require empirically guessing or parametrically sweeping for an exclusion buffer size, NNDM automates this by acting as a distance-targeted performance estimator. It mathematically aligns two cumulative distribution functions:
1.  **$G_{cp}$ (Target Profile):** The distribution of distances from the computational prediction grid (the target map) to the available training data. This represents the empirical geographic leap the final model must actually make.
2.  **$G_{pp}$ (CV Profile):** The distribution of distances from the assessment set ($D_{out}$) to the available analysis set ($D_{in}$) during cross-validation.

**Mechanics:** For each fold, NNDM systematically expands an exclusion buffer, removing the nearest neighbors from $D_{in}$, until the $G_{pp}$ distribution perfectly matches the $G_{cp}$ distribution. This guarantees the spatial cross-validation explicitly mirrors the actual geographic difficulty of the target space, completely removing the need to manually define exclusion parameters.
