# System
import argparse
import os
import warnings
from functools import partial, update_wrapper

# Data science
import pandas as pd
from joblib import parallel_backend

from src.config import CONSTANTS, MODELING
from src.logger import get_logger
from src.utils import (
    fit_test_model,
    optimize_model,
    read_list,
    read_pickle,
    write_pickle,
)

# Constants
DATA_HOME, SEED, TIMEOUT, N_TRIALS, cv = (
    CONSTANTS.DATA_HOME,
    CONSTANTS.SEED,
    MODELING.TIMEOUT,
    MODELING.N_TRIALS,
    MODELING.CV,
)

# Models and configurations
models, random_model_grids, optuna_model_grids, evaluators = (
    MODELING.models,
    MODELING.random_model_grids,
    MODELING.optuna_model_grids,
    MODELING.evaluators,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run ML paper experiment.",
        epilog="""Example of use:

        python predictive_modeling.py --ad  # run predictive modeling for dataset
        python predictive_modeling.py --ra  # run predictive modeling for dataset
        python predictive_modeling.py --ad --optuna # run predictive modeling for dataset, using optuna
        python predictive_modeling.py --ad --random # run predictive modeling for dataset, using random
                """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--ad",
        dest="is_data",
        action="store_true",
    )
    parser.add_argument(
        "--ra",
        dest="is_data",
        action="store_false",
    )
    parser.add_argument(
        "-o",
        "--optuna",
        dest="is_optuna",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--random",
        dest="is_optuna",
        action="store_false",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=TIMEOUT,
        help="Timeout in seconds for Optuna optimization.",
    )
    parser.add_argument(
        "-n",
        "--n_trials",
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help="Number of trials for Optuna optimization, number of iterations in random optimization.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the existing results."
    )
    parser.set_defaults(
        is_data=True,
        debug=False,
        is_optuna=True,
        overwrite=False,
    )
    args = parser.parse_args()

    if args.is_data:
        data_tag = "ad"
        proc_data = pd.read_csv(os.path.join(DATA_HOME, "AD_processed_final.csv"))
        target_col, patient_id = "endpt_lb_easi1_total_score", "patient_id"
        block_list = [
            "ft_sl_actarm",
            "ft_sl_actarmcd",
            "ft_sl_ageu",
            "ft_sl_arm",
            "ft_sl_armcd",
            "ft_sl_domain",
            "ft_sl_dthdtc",
            "ft_sl_ethnic",
            "ft_sl_invid",
            "ft_sl_invnam",
            "ft_sl_rfstdtc",
            "ft_sl_rfendtc",
            "ft_sl_rficdtc",
            "ft_sl_rfpendtc",
            "ft_sl_rfxendtc",
            "ft_sl_rfxstdtc",
            "ft_sl_rowid",
            "ft_sl_dthfl",
            "ft_sl_subjid",
            "ft_sl_studyid",
            "ft_cc_easi1_total_score",
        ] + proc_data.filter(regex="^ft_cc").columns.to_list()
    else:
        data_tag = "ra"
        proc_data = pd.read_csv(os.path.join(DATA_HOME, "RA_processed_final.csv"))
        target_col, patient_id = "endpt_lb_das28__crp_", "patient_id"
        proc_data = proc_data.dropna(
            subset=[target_col, "ft_lb_c_reactive_protein__mg_l_"]
        ).reset_index(drop=True)
        block_list = (
            [
                "ft_sl_actarm",
                "ft_sl_actarmcd",
                "ft_sl_ageu",
                "ft_sl_arm",
                "ft_sl_armcd",
                "ft_sl_domain",
                "ft_sl_dthdtc",
                "ft_sl_dthfl",
                "ft_sl_ethnic",
                "ft_sl_invid",
                "ft_sl_invnam",
                "ft_sl_rfendtc",
                "ft_sl_rficdtc",
                "ft_sl_rfpendtc",
                "ft_sl_rfstdtc",
                "ft_sl_rfxendtc",
                "ft_sl_rfxstdtc",
                "ft_sl_studyid",
                "ft_sl_subjid",
                "ft_sl_siteid",
            ]
            + proc_data.filter(regex="ft_eff").columns.to_list()
            + ["endpt_lb_das28__esr_"]
        )

    opt_type = "optuna" if args.is_optuna else "random"

    # Getting logger
    logger = get_logger(f"logs/{data_tag}_{opt_type}.log")

    logger.info(f"Running {data_tag.upper()} analysis with {opt_type} optimization!")

    # Getting features used for modeling
    features = proc_data.filter(regex="^ft").dropna(axis="columns").columns.to_list()
    features_to_use = [f for f in features if f not in block_list]

    logger.info(
        f"{data_tag.upper()} dataset contains {proc_data.shape[0]} samples and {proc_data.shape[1]} columns.."
    )

    # Getting `X`s
    X_train, X_test = (
        proc_data.loc[proc_data["split"] == "TRAIN", features_to_use],
        proc_data.loc[proc_data["split"] == "TEST", features_to_use],
    )

    # Creating `y`s
    y_train, y_test = (
        proc_data.loc[proc_data["split"] == "TRAIN", target_col],
        proc_data.loc[proc_data["split"] == "TEST", target_col],
    )

    logger.info(
        f"It dataset contains:\n\t"
        f"TRAIN: {X_train.shape[1]} features and {X_train.shape[0]} samples!\n\t"
        f"TEST: {X_test.shape[1]} features and {X_test.shape[0]} samples!\n\t"
    )

    # Read features from files
    sel_fs_dict = {}
    if data_tag == "ra":
        feature_set_tags = [
            "lasso_features_no_rwd_missing",
            "spearman_features_no_rwd_missing",
            "sfs_features_no_rwd_missing",
            "sbs_features_no_rwd_missing",
            "sbs_aic_features_no_rwd_missing",
            "sfs_aic_features_no_rwd_missing",
            "multi_stage_0.05_no_rwd_missing"  # "Multi-stage set"
            "multi_stage_features_0.05_f_num_12_no_rwd_missing",  # "Narrow set"
            "multi_stage_features_0.05_f_num_14_no_rwd_missing",  # "Moderate set"
            "multi_stage_features_0.05_f_num_20_no_rwd_missing_no_mmp3",  # "Wide set without MMP3"
            "multi_stage_features_0.05_f_num_20_no_rwd_missing",  # "Wide set"
            "crp_esr_features",
        ]
    else:
        feature_set_tags = [
            "lasso_features",
            "spearman_features",
            "sfs_features",
            "sbs_features",
            "sfs_aic_features",
            "sbs_aic_features",
            "multi_stage_features_0.05",  # "Multi-stage set"
            "multi_stage_features_0.05_f_num_14",  # "Narrow set"
            "multi_stage_features_0.05_f_num_20",  # "Moderate set"
            "multi_stage_features_0.05_f_num_31",  # "Wide set"
        ]

    for fs_tag in feature_set_tags:
        fn = os.path.join(
            DATA_HOME, f"{data_tag}_selected_features", f"{data_tag}_{fs_tag}.txt"
        )
        sel_fs_dict[fs_tag.replace("_features", "").replace(".", "_")] = read_list(fn)

    # Specify parameters of model optimization
    if opt_type == "optuna":
        opt_config = {
            "timeout": args.timeout,
            "n_trials": args.n_trials,
        }
    else:
        opt_config = {"n_iter": args.n_trials}
    sorted_evals = [f"{t}_{s}" for s in evaluators.keys() for t in ["TRAIN", "TEST"]]

    # To store
    fitted_models = {m: {f: None for f in sel_fs_dict.keys()} for m in models.keys()}
    summary = pd.DataFrame(columns=sel_fs_dict.keys(), index=models.keys())

    # Load the results if they exist
    evals_fn = os.path.join(DATA_HOME, f"{data_tag}_results", opt_type, "evals.pkl")
    if os.path.exists(evals_fn):
        evals = read_pickle(evals_fn)
    else:
        evals = {m: {f: None for f in sel_fs_dict.keys()} for m in models.keys()}

    warnings.simplefilter(action="ignore", category=UserWarning)

    for fs_tag, _ in sel_fs_dict.items():
        # Create a folder with the name of the feature selection method
        if args.debug:
            fs_folder = os.path.join(DATA_HOME, "debug")
        else:
            fs_folder = os.path.join(DATA_HOME, f"{data_tag}_results", opt_type, fs_tag)
        if not os.path.exists(fs_folder):
            os.makedirs(fs_folder)
        logger.info(
            f"CURRENTLY RUNNING FEATURE SELECTION: {fs_tag} ({len(sel_fs_dict[fs_tag])})"
        )
        for m_tag, _ in models.items():
            if (
                not args.overwrite
                and not args.debug
                and evals.get(m_tag, {}).get(fs_tag, None) is not None
            ):
                logger.info(f"Model {m_tag} already fitted, skipping..")
                continue
            logger.info(f"Running tuning for model: {m_tag}")
            fitted_tuner, tuner_best_params, tuner_best_score = optimize_model(
                model_tag=m_tag,
                X_train=X_train,
                y_train=y_train,
                features=sel_fs_dict[fs_tag],
                cv=cv,
                opt_type=opt_type,
                **opt_config,
            )
            logger.info(
                f"   Best params for {m_tag} with {fs_tag}: {tuner_best_params}, score: {tuner_best_score:.3f}"
            )
            logger.info(f"Running training for model: {m_tag}")
            fitted_model, eval_model = fit_test_model(
                model=fitted_tuner.best_estimator_,
                model_params=fitted_tuner.best_params_,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                features=sel_fs_dict[fs_tag],
            )
            fitted_models[m_tag][fs_tag] = fitted_model
            summary.loc[m_tag, fs_tag] = "; ".join(
                [f"{k}: {eval_model[k]:.3f}" for k in sorted_evals]
            )
            evals[m_tag][fs_tag] = eval_model

            # Save model as pickle file
            fn = os.path.join(fs_folder, f"{m_tag}.pkl")
            write_pickle(fitted_model, fn, overwrite=True)
            logger.info(f"Model {m_tag} with {fs_tag} saved as pickle file to {fn}\n")

            if not args.debug:
                # Update evals as pickle
                write_pickle(evals, evals_fn, overwrite=True)

        logger.info(f"Finished running feature selection: {fs_tag}\n\n")

    logger.info("Results:")
    logger.info(summary)
    logger.info("Finished!")


if __name__ == "__main__":
    main()
