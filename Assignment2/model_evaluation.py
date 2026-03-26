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
    # Read and shuffle data
    df = pd.read_csv(path, sep=sep)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Remove duplicates
    print(f"Removing {df.duplicated().sum()} duplicate rows...")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Proportion of correct predictions
    return float(np.mean(y_true == y_pred))


def log_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
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
        self.prob_metrics = prob_metrics or [log_score]
        self.label_encoder = LabelEncoder()
        self.preprocessor = ColumnTransformer(
            [
                # One hot encode categorical features
                ("cat", OneHotEncoder(sparse_output=False), self.categorical),
                # Do not change numerical features
                ("num", "passthrough", self.numerical),
            ]
        )
        self.results = {}

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Fit both encoders and transform data
        y = self.label_encoder.fit_transform(df[self.target])
        X_df = df.drop(columns=[self.target])
        X = self.preprocessor.fit_transform(X_df)
        return X, y

    def _score_fold(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        # Generate model predictions
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)

        # Calculate metric scores
        scores = {}
        for pred_metric in self.pred_metrics:
            scores[pred_metric.__name__] = pred_metric(y_val, y_pred)
        for prob_metric in self.prob_metrics:
            scores[prob_metric.__name__] = prob_metric(y_val, y_prob)
        return scores

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

    """ Unused
    def evaluate_cv(self, df: pd.DataFrame, k: int = 5) -> None:
        # Prepare the dataset
        X, y = self._prepare_data(df)
        self.results = {fn.__name__: [] for fn in self.pred_metrics + self.prob_metrics}

        # For each of k folds
        for train_idx, val_idx in self._make_folds(len(y), k):
            # Calculate and save metric scores
            scores = self._score_fold(
                self._copy_model(), X[train_idx], y[train_idx], X[val_idx], y[val_idx]
            )
            for metric, score in scores.items():
                self.results[metric].append(score)
    """

    def evaluate_nested_cv(
        self,
        df: pd.DataFrame,
        k_outer: int = 5,
        k_inner: int = 5,
        tune_metric: Callable = log_score,  # Metric to minimize
        use_nested: bool = True,
    ) -> None:
        X, y = self._prepare_data(df)
        param_name, param_values = next(iter(self.param_grid.items()))
        self.results = {fn.__name__: [] for fn in self.pred_metrics + self.prob_metrics}

        # For each of k_outer folds
        for outer_train_idx, outer_val_idx in self._make_folds(len(y), k_outer):
            X_tr, y_tr = X[outer_train_idx], y[outer_train_idx]
            X_val, y_val = X[outer_val_idx], y[outer_val_idx]
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
                                X_tr[inner_train_idx],
                                y_tr[inner_train_idx],
                                X_tr[inner_val_idx],
                                y_tr[inner_val_idx],
                            )[tune_metric.__name__]
                        )
                    score = float(np.mean(inner_scores))
                else:
                    # Score parameter directly on outer training set
                    model = self._copy_model(**{param_name: value})
                    score = self._score_fold(model, X_tr, y_tr, X_tr, y_tr)[
                        tune_metric.__name__
                    ]

                # Update best parameter
                if score < best_score:
                    best_score, best_value = score, value

            # Evaluate best parameter on outer validation set
            scores = self._score_fold(
                self._copy_model(**{param_name: best_value}), X_tr, y_tr, X_val, y_val
            )

            # Save metrics
            for fn in self.pred_metrics + self.prob_metrics:
                self.results[fn.__name__].append(scores[fn.__name__])

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

    # Define tunable parameters
    dummy_params = {"strategy": ["prior"]}  # Placeholder param grid
    lr_params = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
    knn_params = {"n_neighbors": [1, 3, 5, 10, 20, 50]}

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

    print("KNN train fold:")
    evaluator = ModelEvaluator(
        KNeighborsClassifier(), target, categorical, numerical, knn_params
    )
    evaluator.evaluate_nested_cv(dataset, use_nested=False)
    evaluator.print_metrics()

    print("KNN nested CV:")
    evaluator = ModelEvaluator(
        KNeighborsClassifier(), target, categorical, numerical, knn_params
    )
    evaluator.evaluate_nested_cv(dataset, use_nested=True)
    evaluator.print_metrics()
