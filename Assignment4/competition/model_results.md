# Model Evaluation Results

Validation uses an anchored `200x300` rectangle grid aligned to the prediction rectangle `data[265:465, 360:660]`. Each rectangle except the prediction rectangle is held out one at a time, and the model trains on the remaining annotated pixels.

<img src="anchored_validation_grid.png" alt="Anchored validation grid" width="540">

Current neural-network settings in `evaluate_models.py`:

| Parameter | Value |
|---|---:|
| Hidden units | `[128, 64]` |
| Activation | ReLU |
| Learning rate | `0.05` |
| Batch size | `512` |
| Regularization | `lambda_=1e-3` |
| Epochs | `5` |
| Coordinate mode | global image coordinates |

## Results

These results were produced by running `evaluate_models.py` with the settings above.

> Pooled log loss is the main validation metric because the competition also scores pixel-wise log loss. Average log loss is also reported because it shows how performance varies across held-out rectangles.

| Model | Features | Avg log loss | Avg accuracy | Pooled log loss | Pooled accuracy | Total time |
|---|---|---:|---:|---:|---:|---:|
| Spectral LR | spectrum | `0.8217 +/- 0.4034` | `0.6850 +/- 0.2080` | `0.7655` | `0.7077` | `67.6s` |
| Spectral NN | spectrum | `0.8634 +/- 0.4073` | `0.6808 +/- 0.1718` | `0.7748` | `0.7172` | `18.9s` |
| Spectral-coordinate NN | spectrum + coordinates | `0.7875 +/- 0.3528` | `0.7118 +/- 0.1601` | `0.6944` | `0.7471` | `21.0s` |
| Local-mean NN | spectrum + 3x3 mean + coordinates | `0.7483 +/- 0.3640` | `0.7357 +/- 0.1634` | `0.6376` | `0.7834` | `27.3s` |
| Multiscale NN | spectrum + 3x3 mean + 5x5 mean + 9x9 mean + coordinates | `0.9443 +/- 0.5383` | `0.7142 +/- 0.1752` | `0.7810` | `0.7663` | `41.3s` |

The best pooled log loss in this run is from the Local-mean NN. The leaderboard result still favors the Spectral-coordinate NN, so the rectangle validation is useful but not perfect.

## Leaderboard Results

These are the public leaderboard scores after submitting the generated prediction files. Lower log loss is better.

| Submitted file | Model | Leaderboard score |
|---|---|---:|
| `spectral_coordinate_nn.npy` | Spectral-coordinate NN | `0.56253` |
| `multiscale_nn.npy` | Multiscale NN | `0.60026` |
| `spectral_lr.npy` | Spectral LR | `0.63675` |
| `local_mean_nn.npy` | Local-mean NN | `0.67365` |
| `spectral_nn.npy` | Spectral NN | `0.79252` |

**Spectral-coordinate NN** has the best leaderboard score, while **Spectral LR** is also competitive. The results do not show a simple trend where more complex models are always better. Spectral information is important, global position helps on the leaderboard, and extra local context depends on the crop.

## Tried Changes

- **Validation:** random stratified cross-validation gave overly optimistic scores, so model selection was moved to anchored rectangle validation. We now use all anchored rectangles except the prediction rectangle.
- **Batch size:** larger batches were faster and gave better log loss than small batches.
- **Regularization:** stronger regularization was kept as a more conservative setting for the held-out rectangles.
- **Epochs:** more epochs did not reliably improve rectangle validation, so the script uses a smaller number of epochs.
- **Coordinates:** global image coordinates worked better than rectangle-relative coordinates.
- **Local context:** adding wider local context helped accuracy in some checks but made log loss less stable.
- **NN layers:** different neural-network layer sizes were discussed, including a one-hidden-layer model, but the current `[128, 64]` setup was kept for the main comparison.
- **CNN models:** CNN or patch-based models were discussed as possible improvements because the task has spatial structure, but they were not implemented in the final evaluation script.
- **Ensembles:** probability ensembles were discussed as a possible improvement for log loss, but the current comparison uses single models only.
