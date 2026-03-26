## Before deduplication:

| Model                    | Evaluation Approach         | Classification Accuracy | Log-score       |
|:-------------------------|:----------------------------|:------------------------|:----------------|
| **DummyClassifier**      | Relative frequency baseline | 0.6081 ± 0.0141         | 1.1658 ± 0.0326 |
| **LogisticRegression**   | Train fold optimization `C` | 0.7357 ± 0.0159         | 0.6711 ± 0.0383 |
| **LogisticRegression**   | Nested cross-validation `C` | 0.7369 ± 0.0156         | 0.6715 ± 0.0387 |
| **KNeighborsClassifier** | Train fold optimization `k` | 0.6933 ± 0.0185         | 6.1686 ± 0.6617 |
| **KNeighborsClassifier** | Nested cross-validation `k` | 0.6986 ± 0.0223         | 0.9698 ± 0.0906 |

## After deduplication with 5 folds:
- Lower accuracy (no seen instances)

| Model                    | Evaluation Approach         | Classification Accuracy | Log-score       |
|:-------------------------|:----------------------------|:------------------------|:----------------|
| **DummyClassifier**      | Relative frequency baseline | 0.6090 ± 0.0154         | 1.1698 ± 0.0321 |
| **LogisticRegression**   | Train fold optimization `C` | 0.7312 ± 0.0159         | 0.6798 ± 0.0346 |
| **LogisticRegression**   | Nested cross-validation `C` | 0.7308 ± 0.0158         | 0.6795 ± 0.0347 |
| **KNeighborsClassifier** | Train fold optimization `k` | 0.6833 ± 0.0147         | 6.3590 ± 0.4812 |
| **KNeighborsClassifier** | Nested cross-validation `k` | 0.6951 ± 0.0180         | 0.9999 ± 0.0926 |

## With 10 outer folds:
- Higher accuracy (more training data)
- Higher variance (fewer instances in test folds)

| Model                    | Evaluation Approach         | Classification Accuracy | Log-score       |
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
