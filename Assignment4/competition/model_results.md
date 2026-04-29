# Model Evaluation Results

Validation uses the anchored `200x300` rectangle grid aligned to the competition rectangle. The competition cell is skipped, and cells `1, 3, 5, 7, 9, 11, 13, 15` are held out one at a time.

Current neural-network settings in `evaluate_models.py`:

| Parameter | Value |
|---|---:|
| Hidden units | `[128, 64]` |
| Activation | ReLU |
| Learning rate | `0.05` |
| Batch size | `512` |
| Regularization | `lambda_=1e-3` |
| Epochs | `10` |
| Coordinate mode | global image coordinates |

## Results

| Model | Features | Avg log loss | Avg accuracy | Pooled log loss | Pooled accuracy | Total time |
|---|---|---:|---:|---:|---:|---:|
| Logistic regression | spectrum | `0.8276 +/- 0.3852` | `0.7271 +/- 0.1643` | `0.8027` | `0.7212` | `36.9s` |
| Spectral NN | spectrum | `0.8574 +/- 0.3889` | `0.7157 +/- 0.1698` | `0.7637` | `0.7466` | `17.7s` |
| Spectral-coordinate NN | spectrum + coordinates | `0.8041 +/- 0.4152` | `0.7392 +/- 0.1769` | `0.6966` | `0.7743` | `17.3s` |
| Local-mean NN | spectrum + 3x3 mean + coordinates | `0.8884 +/- 0.5914` | `0.7395 +/- 0.1927` | `0.7129` | `0.7885` | `25.9s` |
| Multiscale NN | spectrum + 3x3 mean + 5x5 mean + 9x9 mean + coordinates | `1.1716 +/- 0.8909` | `0.7050 +/- 0.2357` | `0.8319` | `0.7834` | `32.3s` |

Best current model by pooled log loss is the Spectral-coordinate NN.

## Tried Changes

- Random stratified validation gave much more optimistic scores, so model selection was moved to anchored rectangle validation.
- `batch_size=128` was worse and slower than `256`; `batch_size=512` improved log loss and speed.
- `lambda_=1e-4` was slightly best in one small sweep, but `lambda_=1e-3` was kept as a stronger regularized setting.
- `100` epochs with stronger regularization, including `lambda_=1e-2`, did not improve held-out rectangle log loss.
- Rectangle-relative coordinates hurt performance compared with global image coordinates.
- Adding `9x9` context to the multiscale model improved accuracy in some checks but worsened log loss and stability.
- For the Local-mean NN, fewer epochs helped substantially: `10` epochs beat `25` and `50` epochs on pooled log loss.
- Potential improvement: try `5` epochs for the Local-mean NN; it achieved lower pooled log loss in a separate check.
- Potential improvement: try a one-hidden-layer `[128]` ANN for the Spectral-coordinate NN; it achieved lower pooled log loss in an architecture check.
