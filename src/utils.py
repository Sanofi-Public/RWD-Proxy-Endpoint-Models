"""Utilities for the project."""

import os
import pickle
import warnings
from operator import attrgetter
from typing import List, Optional, Union

import numpy as np
import optuna
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVR

from .config import CONSTANTS, MODELING

SEED = CONSTANTS.SEED
models = MODELING.models
random_model_grids = MODELING.random_model_grids
optuna_model_grids = MODELING.optuna_model_grids
evaluators = MODELING.evaluators


def read_pickle(file_path):
    """Reads a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def write_pickle(obj, file_path, overwrite=False):
    """Writes an object to a pickle file."""
    if not overwrite and os.path.exists(file_path):
        raise FileExistsError(f"{file_path} already exists.")

    if os.path.exists(file_path):
        warnings.warn(f"{file_path} already exists. Overwriting.")
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def read_list(file_path):
    """Reads a list from a file."""
    with open(file_path) as f:
        return f.read().splitlines()


def write_list(obj, file_path, overwrite=False):
    """Writes a list to a file."""
    if not overwrite and os.path.exists(file_path):
        raise FileExistsError(f"{file_path} already exists.")

    if os.path.exists(file_path):
        warnings.warn(f"{file_path} already exists. Overwriting.")
    with open(file_path, "w") as f:
        f.write("\n".join(obj))


def optimize_model(
    model_tag,
    X_train,
    y_train,
    features,
    cv,
    groups=None,
    opt_type="random",
    **opt_config,
):
    """A helper function to optimize the passed model hyperparams."""

    if opt_type == "random":
        tuner = RandomizedSearchCV(
            models[model_tag],
            param_distributions=random_model_grids[model_tag],
            cv=cv,
            random_state=SEED,
            n_jobs=-1,
            **opt_config,
        ).fit(X_train[features], y_train, groups=groups)
    elif opt_type == "optuna":
        tuner = optuna.integration.OptunaSearchCV(
            models[model_tag],
            param_distributions=optuna_model_grids[model_tag],
            cv=cv,
            random_state=SEED,
            n_jobs=-1,
            verbose=0,
            **opt_config,
        ).fit(X_train[features], y_train, groups=groups)
    else:
        raise ValueError(f"Unknown optimization type: {opt_type}")

    return tuner, tuner.best_params_, tuner.best_score_


def fit_test_model(model, model_params, X_train, X_test, y_train, y_test, features):
    """A helper function to refit the model on the whole "TRAIN" set and obtain performance
    metrics."""

    if model.__class__.__name__ == "LinearRegression":
        model_init = LinearRegression
    elif model.__class__.__name__ == "GradientBoostingRegressor":
        model_init = GradientBoostingRegressor
    elif model.__class__.__name__ == "RandomForestRegressor":
        model_init = RandomForestRegressor
    elif model.__class__.__name__ == "LinearSVR":
        model_init = LinearSVR
    elif model.__class__.__name__ == "ExplainableBoostingRegressor":
        model_init = ExplainableBoostingRegressor

    optimized_model = model_init(**model_params)
    refit_model = optimized_model.fit(X_train[features], y_train)

    y_pred_train, y_pred_test = (
        refit_model.predict(X_train[features]),
        refit_model.predict(X_test[features]),
    )

    evaluation = pd.Series(
        [
            func(y_train, y_pred_train) if s == "TRAIN" else func(y_test, y_pred_test)
            for s in ["TRAIN", "TEST"]
            for name, func in evaluators.items()
        ],
        index=[
            f"{s}_{metric}" for s in ["TRAIN", "TEST"] for metric in evaluators.keys()
        ],
    )

    return refit_model, evaluation


def importance_getter_k(estimator, k):
    """A helper function that transforms the calculated feature importances by zero-ing all
    importances except for top k ones. That way we will force the selector to pick only
    top k features. This is useful if we do not want to specify threshold value, but specify the
    number of features to pick."""

    if hasattr(estimator, "coef_"):
        getter = attrgetter("coef_")
    elif hasattr(estimator, "feature_importances_"):
        getter = attrgetter("feature_importances_")

    importances = getter(estimator)
    if importances.ndim == 1:
        importances = np.abs(importances)
    else:
        importances = np.linalg.norm(importances, axis=0, ord=1)
    if k == "all":
        k = len(importances)
    k = min(k, len(importances))
    thresh = importances[np.argsort(importances)[::-1][k - 1]]
    importances[importances < thresh] = 0

    return importances
