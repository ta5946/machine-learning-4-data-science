import os
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing import Callable, Dict, List, Optional, Any, Tuple, cast

warnings.filterwarnings("ignore", category=ConvergenceWarning)
random_state = 0


def read_shuffle_csv(path: str, sep: str = ";") -> pd.DataFrame:
    # Read data
    df = pd.read_csv(path, sep=sep)
    # Assign IDs
    df.insert(0, "id", range(len(df)))
    # Shuffle data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Remove duplicates
    print(
        f"Removing {df.duplicated(subset=df.columns.difference(['id'])).sum()} duplicate rows..."
    )
    df = df.drop_duplicates(subset=df.columns.difference(["id"])).reset_index(drop=True)
    return df


def classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Proportion of correct predictions
    return float(np.mean(y_true == y_pred))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Avoid log(0) = -inf
    y_prob = np.clip(y_prob, 1e-15, 1.0)
    true_class_probs = y_prob[np.arange(len(y_true)), y_true]

    # Negative log likelihood of correct class
    return float(-np.mean(np.log(true_class_probs)))


class ModelEvaluator:

    def __init__(
        self,
        model: BaseEstimator,
        target: str,
        categorical: List[str],
        numerical: List[str],
        param_grid: Optional[
            Dict[str, List[Any]]
        ] = None,  # 1D grid of parameter values
        pred_metrics: Optional[List[Callable]] = None,
        prob_metrics: Optional[List[Callable]] = None,
    ):
        self.model = model
        self.target = target
        self.categorical = categorical
        self.numerical = numerical
        self.param_grid = param_grid
        self.pred_metrics = pred_metrics or [classification_accuracy]
        self.prob_metrics = prob_metrics or [log_loss]
        self.label_encoder = LabelEncoder()
        self.preprocessor = ColumnTransformer(
            [
                # One hot encode categorical features
                (
                    "cat",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    self.categorical,
                ),  # Avoid runtime errors
                # Do not change numerical features
                ("num", "passthrough", self.numerical),
            ]
        )
        self.results = {}
        self.predictions = pd.DataFrame()

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        # Fit label encoder on whole dataframe for consistent integer mapping
        # Keep raw features and handle encoding in _score_fold to avoid data leakage
        y = self.label_encoder.fit_transform(df[self.target])
        X_df = df.drop(columns=[self.target])
        return X_df, y

    def _score_fold(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        # Fit preprocessor on training data and transform validation data
        preprocessor = clone(self.preprocessor)
        X_train_enc = preprocessor.fit_transform(X_train)
        X_val_enc = preprocessor.transform(X_val)

        # Generate model predictions
        model.fit(X_train_enc, y_train)
        y_pred = model.predict(X_val_enc)
        y_prob = model.predict_proba(X_val_enc)

        # Calculate metric scores
        scores = {}
        for pred_metric in self.pred_metrics:
            scores[pred_metric.__name__] = pred_metric(y_val, y_pred)
        for prob_metric in self.prob_metrics:
            scores[prob_metric.__name__] = prob_metric(y_val, y_prob)
        return (
            scores,
            y_pred,
            y_prob,
        )  # Also return generated predictions and probabilities

    def _make_folds(self, n: int, k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        # Create 2D array chunks of indices
        splits = np.array_split(np.arange(n), k)
        folds = []

        # For each of k folds
        for i in range(k):
            # Use chunk i for validation and the rest for training
            val_idx = splits[i]
            train_idx = np.concatenate(splits[:i] + splits[i + 1 :])
            folds.append((train_idx, val_idx))
        return folds

    def _copy_model(self, **kwargs) -> BaseEstimator:
        # Create unfitted clone
        model = cast(BaseEstimator, clone(self.model))

        # Override parameters
        if kwargs:
            model.set_params(**kwargs)
        return model

    def evaluate_nested_cv(
        self,
        df: pd.DataFrame,
        k_outer: int = 5,
        k_inner: int = 5,
        tune_metric: Callable = log_loss,  # Metric to minimize
        use_nested: bool = True,
    ) -> None:
        X, y = self._prepare_data(df)
        param_name, param_values = next(iter(self.param_grid.items()))
        self.results = {fn.__name__: [] for fn in self.pred_metrics + self.prob_metrics}
        # Collect instance predictions from outer folds for later analysis
        pred_rows = []

        # For each of k_outer folds
        for outer_train_idx, outer_val_idx in self._make_folds(len(y), k_outer):
            X_tr, y_tr = X.iloc[outer_train_idx], y[outer_train_idx]
            X_val, y_val = X.iloc[outer_val_idx], y[outer_val_idx]
            best_value, best_score = None, np.inf

            # Try each parameter value from the grid
            for value in param_values:
                if use_nested:
                    # Perform inner CV to score parameter
                    inner_scores = []
                    for inner_train_idx, inner_val_idx in self._make_folds(
                        len(y_tr), k_inner
                    ):
                        inner_scores.append(
                            self._score_fold(
                                self._copy_model(**{param_name: value}),
                                X_tr.iloc[inner_train_idx],
                                y_tr[inner_train_idx],
                                X_tr.iloc[inner_val_idx],
                                y_tr[inner_val_idx],
                            )[0][tune_metric.__name__]
                        )
                    score = float(np.mean(inner_scores))
                else:
                    # Score parameter directly on outer training set
                    score = self._score_fold(
                        self._copy_model(**{param_name: value}),
                        X_tr,
                        y_tr,
                        X_tr,
                        y_tr,
                    )[0][tune_metric.__name__]

                # Update best parameter
                if score < best_score:
                    best_score, best_value = score, value

            # Evaluate best parameter on outer validation set
            scores, y_pred, y_prob = self._score_fold(
                self._copy_model(**{param_name: best_value}), X_tr, y_tr, X_val, y_val
            )
            # Save metrics
            for fn in self.pred_metrics + self.prob_metrics:
                self.results[fn.__name__].append(scores[fn.__name__])

            # Save fold instances
            fold_preds = df.iloc[outer_val_idx].copy().reset_index(drop=True)
            fold_preds["y_true"] = self.label_encoder.inverse_transform(y_val)
            # Extract predictions
            fold_preds["y_pred"] = self.label_encoder.inverse_transform(y_pred)
            fold_preds["best_param_value"] = best_value
            fold_preds["model"] = self.model.__class__.__name__

            # For each encoded class
            for i, cls in enumerate(self.label_encoder.classes_):
                # Extract probability
                fold_preds[f"prob_{cls}"] = y_prob[:, i]
            pred_rows.append(fold_preds)

        # Combine predictions from all folds
        self.predictions = pd.concat(pred_rows, ignore_index=True)

    def save_predictions(self, use_nested: bool, folder: str = "predictions") -> None:
        # Save to predictions/model_tuning.csv
        os.makedirs(folder, exist_ok=True)
        model_name = self.model.__class__.__name__
        tuning = "nested_cv" if use_nested else "train_fold"
        path = os.path.join(folder, f"{model_name}_{tuning}.csv")
        self.predictions.to_csv(path, index=False)
        print(f"Saved predictions to {path}")

    def print_metrics(self) -> None:
        # Get base model name
        row = {"model": self.model.__class__.__name__}

        # For each metric
        for metric, scores in self.results.items():
            # Get score mean +- standard deviation
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"

        # Print table without index
        print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    # Import models
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    # Load the dataset
    dataset = read_shuffle_csv("dataset.csv")
    print(f"Dataset shape: {dataset.shape}")

    # Define target and features
    target = "ShotType"
    categorical = ["Competition", "PlayerType", "Movement"]
    numerical = ["Transition", "TwoLegged", "Angle", "Distance"]

    # Print the number of identical rows with different labels
    duplicate_count = (
        dataset.drop(columns=["id", target]).duplicated(keep=False).sum()
    ) / 2
    print(f"Number of rows conflicting labels: {int(duplicate_count)}")

    # Define tunable parameters
    dummy_params = {"strategy": ["prior"]}  # Placeholder param grid
    lr_params = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
    knn_params = {"n_neighbors": [1, 3, 5, 10, 20, 50]}  # sqrt(5000 instances)

    # Evaluate each model and print metrics
    print("Relative frequency baseline:")
    evaluator = ModelEvaluator(
        DummyClassifier(random_state=random_state),
        target,
        categorical,
        numerical,
        dummy_params,
    )
    evaluator.evaluate_nested_cv(dataset, use_nested=False)
    evaluator.print_metrics()
    evaluator.save_predictions(use_nested=False)

    print("LR train fold:")
    evaluator = ModelEvaluator(
        LogisticRegression(max_iter=500, random_state=random_state),
        target,
        categorical,
        numerical,
        lr_params,
    )
    evaluator.evaluate_nested_cv(dataset, use_nested=False)
    evaluator.print_metrics()
    evaluator.save_predictions(use_nested=False)

    print("LR nested CV:")
    evaluator = ModelEvaluator(
        LogisticRegression(max_iter=500, random_state=random_state),
        target,
        categorical,
        numerical,
        lr_params,
    )
    evaluator.evaluate_nested_cv(dataset, use_nested=True)
    evaluator.print_metrics()
    evaluator.save_predictions(use_nested=True)

    print("KNN train fold:")
    evaluator = ModelEvaluator(
        KNeighborsClassifier(), target, categorical, numerical, knn_params
    )
    evaluator.evaluate_nested_cv(dataset, use_nested=False)
    evaluator.print_metrics()
    evaluator.save_predictions(use_nested=False)

    print("KNN nested CV:")
    evaluator = ModelEvaluator(
        KNeighborsClassifier(), target, categorical, numerical, knn_params
    )
    evaluator.evaluate_nested_cv(dataset, use_nested=True)
    evaluator.print_metrics()
    evaluator.save_predictions(use_nested=True)
