from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
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
EXAMPLES_TO_INTERPRET = pd.DataFrame(
    [
        {"x1": 0.4, "x2": -0.4, "x3": 1},
        {"x1": 0.2, "x2": -0.4, "x3": 1},
        {"x1": 0.4, "x2": -0.4, "x3": 2},
        {"x1": 0.4, "x2": 0.2, "x3": 2},
    ]
)


def make_one_hot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def load_data(path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    x = data[CONTINUOUS_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]
    return x, y


def make_preprocessor() -> ColumnTransformer:
    # Keep preprocessing identical for all models, so scores and explanations
    # are easier to compare.
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_FEATURES),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_models() -> dict[str, Pipeline]:
    # Use one linear baseline and a few common non-linear regressors for tabular data.
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
    kernel_width: float | None = None,
) -> pd.DataFrame:
    if len(instance) != 1:
        raise ValueError("LIME explains one instance at a time.")

    rng = np.random.default_rng(RANDOM_STATE)
    instance_row = instance.iloc[0]

    # LIME explains one point by creating many artificial points around it.
    # The original instance is kept as the first row of this local dataset.
    perturbed_samples = pd.DataFrame(
        index=range(num_samples), columns=training_data.columns
    )
    perturbed_samples.iloc[0] = instance_row

    # Numeric features are sampled from the training distribution, which keeps
    # the artificial points in a plausible range.
    for feature in CONTINUOUS_FEATURES:
        mean = training_data[feature].mean()
        std = training_data[feature].std()
        perturbed_samples.loc[1:, feature] = rng.normal(
            mean, std, size=num_samples - 1
        )

    # Categorical values are sampled in the same proportions as in the dataset,
    # so common categories appear more often in the local sample.
    for feature in CATEGORICAL_FEATURES:
        category_counts = training_data[feature].value_counts(normalize=True).sort_index()
        perturbed_samples.loc[1:, feature] = rng.choice(
            category_counts.index,
            size=num_samples - 1,
            p=category_counts.values,
        )

    perturbed_samples = perturbed_samples.astype(training_data.dtypes.to_dict())

    # The original model gives the target values that the local explanation tries
    # to mimic. LIME does not use the true y values here.
    predictions = model.predict(perturbed_samples)

    # The local model uses simpler features than the original data: standardized
    # numeric values and binary category-match indicators.
    surrogate_features = pd.DataFrame(index=perturbed_samples.index)
    interpretable_instance = []
    feature_names = []

    for feature in CONTINUOUS_FEATURES:
        mean = training_data[feature].mean()
        std = training_data[feature].std()
        surrogate_features[feature] = (perturbed_samples[feature] - mean) / std
        interpretable_instance.append((instance_row[feature] - mean) / std)
        feature_names.append(feature)

    for feature in CATEGORICAL_FEATURES:
        category = instance[feature].iloc[0]
        feature_name = f"{feature}={category}"
        surrogate_features[feature_name] = (
            perturbed_samples[feature] == category
        ).astype(int)
        interpretable_instance.append(1)
        feature_names.append(feature_name)

    interpretable_instance = np.array(interpretable_instance, dtype=float)
    surrogate_values = surrogate_features.to_numpy(dtype=float)

    # Closer artificial points should describe the model behavior near this
    # instance better than far away points.
    distances = np.linalg.norm(surrogate_values - interpretable_instance, axis=1)

    if kernel_width is None:
        kernel_width = np.sqrt(len(feature_names)) * 0.75

    # Turn distances into weights: close points get large weights, far points
    # get small weights.
    weights = np.exp(-(distances**2) / (kernel_width**2))

    # Fit a simple weighted linear model on model predictions. Its coefficients
    # are the LIME explanation for this one prediction.
    surrogate_model = Lasso(alpha=0.001, max_iter=10000, random_state=RANDOM_STATE)
    surrogate_model.fit(surrogate_values, predictions, sample_weight=weights)

    return pd.DataFrame(
        {
            "feature": feature_names,
            "lime_weight": surrogate_model.coef_,
        }
    ).sort_values("lime_weight", key=abs, ascending=False)


def explain_regression(model: Pipeline) -> pd.DataFrame:
    # For the linear model, inspect the coefficients after preprocessing.
    preprocessor = model.named_steps["preprocess"]
    regression_model = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = regression_model.coef_

    return pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    ).sort_values("coefficient", key=abs, ascending=False)


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
    training_data, _ = load_data()
    print("Model evaluation:")
    print(evaluation_results.to_string(index=False))

    # Interpret two strong non-linear models and the linear model as a baseline.
    models_to_interpret = ["gradient_boosting", "mlp", "linear_regression"]

    for model_name in models_to_interpret:
        model = models[model_name]
        example_predictions = EXAMPLES_TO_INTERPRET.copy()
        example_predictions["prediction"] = model.predict(EXAMPLES_TO_INTERPRET)

        print(f"\nExamples to interpret using {model_name}:")
        print(example_predictions.to_string(index=False))

        print(f"\nLIME explanations for {model_name}:")
        for example_index in EXAMPLES_TO_INTERPRET.index:
            lime_explanation = explain_lime(
                model,
                EXAMPLES_TO_INTERPRET.loc[[example_index]],
                training_data,
            )
            print(f"\nExample {example_index + 1}:")
            print(lime_explanation.to_string(index=False))

        if "regression" in model_name:
            # Linear regression gives us direct coefficients to compare with LIME.
            print(f"\nRegression coefficients for {model_name}:")
            print(explain_regression(model).to_string(index=False))
