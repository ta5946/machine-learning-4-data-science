from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = "toydataset.csv"
TEST_SIZE = 0.2

CONTINUOUS_FEATURES = ["x1", "x2"]
CATEGORICAL_FEATURES = ["x3"]
TARGET = "y"


def make_one_hot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def load_data(path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    x = data[CONTINUOUS_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]
    return x, y


def make_preprocessor() -> ColumnTransformer:
    # Keep preprocessing identical for all models, so their scores and later
    # explanations are easier to compare.
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_FEATURES),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_models() -> dict[str, Pipeline]:
    # A simple set of regressors: one linear baseline plus a few common
    # non-linear models for tabular data.
    return {
        "linear_regression": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        alpha=0.001,
                        early_stopping=True,
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=-1),
                ),
            ]
        ),
    }


def evaluate_model(
    model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    predictions = model.predict(x_test)
    return {  # type: ignore[return-value]
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


def explain_lime(
    model: Pipeline,
    instance: pd.DataFrame,
    training_data: pd.DataFrame,
    num_samples: int = 1000,
    kernel_width: float = 0.75,
) -> pd.DataFrame:
    # TODO: Implement LIME for one prediction.
    raise NotImplementedError


def train_models() -> tuple[dict[str, Pipeline], pd.DataFrame]:
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = make_models()
    evaluation_rows = []

    # Train every model on the same split and collect the test-set scores.
    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        evaluation_rows.append({"model": name, **metrics})

    evaluation_results = pd.DataFrame(evaluation_rows).sort_values(
        "r2", ascending=False
    )
    return models, evaluation_results


if __name__ == "__main__":
    models, evaluation_results = train_models()
    print(evaluation_results.to_string(index=False))
