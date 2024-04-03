# Proxy endpoints - Bridging Clinical Trials and Real World Data

Disease severity scores, or endpoints, are routinely measured during Randomized Controlled Trials
(RCTs) to closely monitor the effect of treatment on a disease. In real-world clinical practice, although
a larger set of patients is observed, the RCT endpoints are often not captured, which makes it hard
to integrate real-world data (RWD) with RCT data to evaluate treatment efficacy and the quality
of patient care. To overcome this challenge, we developed an ensemble technique which learns
proxy models of disease endpoints in RWD. Using a multi-stage learning framework applied to RCT
data, we first identify features considered significant drivers of disease available within RWD. To
create endpoint proxy models, we use Explainable Boosting Machines (EBMs) which allow for both
end-user interpretability and modeling of non-linear relationships. We demonstrate our approach
on two diseases, rheumatoid arthritis (RA) and atopic dermatitis (AD). As we show, our combined
feature selection and prediction method achieves good results for both diseases, opening the door to
much wider use of RWD data in treatment evaluation. In most cases our methods improve upon prior
methods proposed to predict disease severity.

## Install

To install the required libraries, run the following command:

```bash
# Install conda environment
conda create -n proxy_paper python=3.8 -y  # we are using python 3.8
conda activate proxy_paper

# Install dependencies
pip install -r requirements.txt
```

## Running

To run the pipeline, please first put the datasets in the `DATA_HOME` folder (look in [`config.py`](src/config.py) and [analytical](notebooks/multi_stage_feature_selection.ipynb) [notebooks](notebooks/benchmark_feature_selection.ipynb)) and create an environment with libraries from [`requirements.txt`](./requirements.txt) as described above. Then, you can run the notebooks and scripts in the following order:

1. Run the [`multi_stage_feature_selection.ipynb`](notebooks/multi_stage_feature_selection.ipynb) to identify proxy feature candidates
2. Run the predictive modeling pipeline in the [`predictive_modeling.py`](scripts/predictive_modeling.py) to train and evaluate the predictive proxy models on AD and RA datasets
   ```
   # run predictive modeling for AD dataset, using optuna framework
   python scripts/predictive_modeling.py --ad --optuna

   # run predictive modeling for RA dataset, using optuna framework
   python scripts/predictive_modeling.py --ra --optuna
   ```
3. (optional) Run the [`benchmark_feature_selection.ipynb`](notebooks/benchmark_feature_selection.ipynb) to compare the performance of trained models with other feature selection methods

All outputs are stored in defined `DATA_HOME`:

```bash
`DATA_HOME`
├── ad_results  # all AD results from optuna / random runs
│   ├── optuna
│   ├── random
├── ad_selected_features  # all AD feature sets
│   ├── ad_lasso_features.txt
│   ├── ad_multi_stage_features_0.05.txt
│   ├── ad_multi_stage_features_0.05_f_num_14.txt
│   ├── ad_multi_stage_features_0.05_f_num_20.txt
│   ├── ad_multi_stage_features_0.05_f_num_31.txt
│   ├── ad_sbs_aic_features.txt
│   ├── ad_sbs_features.txt
│   ├── ad_sfs_aic_features.txt
│   ├── ad_sfs_features.txt
│   └── ad_spearman_features.txt
├── ra_results  # all RA results from optuna / random runs
│   ├── optuna
│   └── random
├── ra_selected_features  # all RA feature sets
│   ├── ra_crp_esr_features.txt
│   ├── ra_lasso_features_no_rwd_missing.txt
│   ├── ra_multi_stage_features_0.05_f_num_12_no_rwd_missing.txt
│   ├── ra_multi_stage_features_0.05_f_num_14_no_rwd_missing.txt
│   ├── ra_multi_stage_features_0.05_f_num_20_no_rwd_missing.txt
│   ├── ra_multi_stage_features_0.05_f_num_20_no_rwd_missing_no_mmp3.txt
│   ├── ra_multi_stage_features_0.05_no_rwd_missing.txt
│   ├── ra_sbs_aic_features_no_rwd_missing.txt
│   ├── ra_sbs_features_no_rwd_missing.txt
│   ├── ra_sfs_aic_features_no_rwd_missing.txt
│   ├── ra_sfs_features_no_rwd_missing.txt
│   └── ra_spearman_features_no_rwd_missing.txt
├── AD_final.csv  # Input AD data
├── AD_ppl_final.pkl  # AD processing pipeline object storing imputation, encoding, normalization
├── AD_processed_final.csv  # Processed AD data
├── RA_final.csv  # Input RA data
├── RA_ppl_final.pkl  # RA processing pipeline object storing imputation, encoding, normalization
├── RA_processed_final.csv  # Processed RA data
```

*While the code is centered around rheumatoid arthritis (RA) and atopic dermatitis (AD) datasets, this code base may be extended to other clinical trial datasets.*

## Repository

The repository contains following files that you should pay attention to:

[`notebooks`](notebooks/)

- [`multi_stage_feature_selection.ipynb`](notebooks/multi_stage_feature_selection.ipynb) contains the full **multi stage feature selection pipeline** used to identify proxy feature candidates in an automated fashion
- [`benchmark_feature_selection.ipynb`](notebooks/benchmark_feature_selection.ipynb) contains the **benchmarking feature selection pipeline** used to verify the results of the multi stage feature selection (i.e. `Spearman`-based, `Lasso`-based, other methods for feature selection)

[`scripts`](scripts/)

- [`predictive_modeling.py`](scripts/predictive_modeling.py) contains the code to train and evaluate the predictive models

[`src`](src/)

- [`config.py`](src/config.py) contains the configuration of the project
- [`logger.py`](src/logger.py) contains the logger
- [`metrics.py`](src/metrics.py) contains the metrics used to evaluate the predictive models
- [`utils.py`](src/utils.py) contains the utility functions used in the notebooks and scripts

## Datasets

The datasets used in this project are not publicly available. However, the code provided in this repository can be used to reproduce the results on similar datasets. The rheumatoid arthritis (RA) data was aggregated from two clinical trials [NCT02309359](https://classic.clinicaltrials.gov/ct2/show/NCT02309359) and [NCT02287922](https://classic.clinicaltrials.gov/ct2/show/NCT02287922) and is internal to Sanofi. Atopic dermatitis (AD) dataset consisted of two clinical trials [NCT03569293](https://classic.clinicaltrials.gov/ct2/show/NCT03569293) and [NCT03568318](https://classic.clinicaltrials.gov/ct2/show/NCT03568318) which were provided by Datacelerate.

## Contacts

For any inquiries please contact:

- Brandon Rufino - brandon.rufino@sanofi.com
- Maxim Kryukov - maksim.kriukov@sanofi.com

## Code references

- Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2). ([Publisher link](https://www.nature.com/articles/s41586-020-2649-2)).
- The pandas development team. pandas-dev/pandas: Pandas, February 2020. DOI: [10.5281/zenodo.3509134](https://doi.org/10.5281/zenodo.3509134)
- [Scikit-learn: Machine Learning in Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
- Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16). Association for Computing Machinery, New York, NY, USA, 785–794. https://doi.org/10.1145/2939672.2939785
- "InterpretML: A Unified Framework for Machine Learning Interpretability" (H. Nori, S. Jenkins, P. Koch, and R. Caruana 2019)
- Raschka, Sebastian (2018) MLxtend: Providing machine learning and data science utilities and extensions to Python's scientific computing stack. J Open Source Softw 3(24).
- Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019.
Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.

## License

Permission is hereby granted, free of charge, for academic research purposes only and for non-commercial uses only, to any person from academic research or non-profit organizations obtaining a copy of this software and associated documentation files (the "Software"), to use, copy, modify, or merge the Software, subject to the following conditions: this permission notice shall be included in all copies of the Software or of substantial portions of the Software. For purposes of this license, “non-commercial use” excludes uses foreseeably resulting in a commercial benefit. To use this software for other purposes (such as the development of a commercial product, including but not limited to software, service, or pharmaceuticals), please contact SANOFI. All other rights are reserved. The Software is provided “as is”, without warranty of any kind, express or implied, including the warranties of noninfringement.
