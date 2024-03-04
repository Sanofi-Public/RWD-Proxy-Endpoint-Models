"""Project configuration."""

import optuna
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import LinearSVR

from .metrics import spearman_score


class CONSTANTS:
    """General configuration."""

    # Data
    DATA_HOME = ""

    # Fixing random experiments
    SEED = 42


class MODELING:
    """Model-specific configuration."""

    # Experiment runtime
    TIMEOUT = 60 * 36  # 36 minutes per experiment (40 experiments running for 24 hours)
    N_TRIALS = 1000  # 1000 trials per experiment
    CV = 5  # 5-fold cross-validation

    models = {
        "linear": LinearRegression(),
        "grad_boost": GradientBoostingRegressor(),
        "rf": RandomForestRegressor(),
        "svm": LinearSVR(),
        "ebm": ExplainableBoostingRegressor(),
    }

    # RandomizedSearchCV grids
    random_model_grids = {
        "linear": {},
        "grad_boost": {
            "n_estimators": [10, 25, 50, 100, 200, 500, 1000],
            "random_state": [CONSTANTS.SEED],
            "min_samples_split": [2, 5, 10],
        },
        "rf": {
            "n_estimators": [10, 25, 50, 100, 200, 500, 1000],
            "random_state": [CONSTANTS.SEED],
            "min_samples_split": [2, 5, 10],
            "n_jobs": [-1],
        },
        "svm": {
            "C": [0.001, 0.01, 0.1, 1, 10],
            "dual": [False],
            "random_state": [CONSTANTS.SEED],
            "loss": ["squared_epsilon_insensitive"],
        },
        "ebm": {
            "random_state": [CONSTANTS.SEED],
            "interactions": [0, 2, 10],
            "max_bins": [128, 256, 512],
            "max_leaves": [3, 4, 5],
            "n_jobs": [-1],
        },
    }

    # Optuna grids
    optuna_model_grids = {
        "linear": {},
        "grad_boost": {
            "n_estimators": optuna.distributions.IntDistribution(10, 1000),
            "random_state": optuna.distributions.CategoricalDistribution(
                [CONSTANTS.SEED]
            ),
            "min_samples_split": optuna.distributions.IntDistribution(2, 10),
            "max_depth": optuna.distributions.IntDistribution(2, 10),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.1),
            "max_features": optuna.distributions.FloatDistribution(0.01, 1.0),
        },
        "rf": {
            "n_estimators": optuna.distributions.IntDistribution(10, 1000),
            "random_state": optuna.distributions.CategoricalDistribution(
                [CONSTANTS.SEED]
            ),
            "min_samples_split": optuna.distributions.IntDistribution(2, 10),
            "n_jobs": optuna.distributions.CategoricalDistribution([-1]),
            "max_depth": optuna.distributions.IntDistribution(2, 10),
            "max_features": optuna.distributions.FloatDistribution(0.01, 1.0),
        },
        "svm": {
            "C": optuna.distributions.FloatDistribution(1e-10, 1e10, log=True),
            "dual": optuna.distributions.CategoricalDistribution([False]),
            "random_state": optuna.distributions.CategoricalDistribution(
                [CONSTANTS.SEED]
            ),
            "loss": optuna.distributions.CategoricalDistribution(
                ["squared_epsilon_insensitive"]
            ),
        },
        "ebm": {
            "random_state": optuna.distributions.CategoricalDistribution(
                [CONSTANTS.SEED]
            ),
            "interactions": optuna.distributions.IntDistribution(1, 10),
            "outer_bags": optuna.distributions.IntDistribution(2, 25),
            "inner_bags": optuna.distributions.IntDistribution(2, 25),
            "max_bins": optuna.distributions.IntDistribution(2, 512),
            "max_leaves": optuna.distributions.IntDistribution(2, 5),
            "early_stopping_tolerance": optuna.distributions.FloatDistribution(
                0.0001, 0.01
            ),
            "early_stopping_rounds": optuna.distributions.IntDistribution(1, 100),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 10),
            "n_jobs": optuna.distributions.CategoricalDistribution([-1]),
        },
    }

    # Evaluators
    evaluators = {
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "spearman_score": spearman_score,
    }
