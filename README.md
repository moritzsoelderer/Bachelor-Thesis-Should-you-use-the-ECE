# Project Overview

This repository contains the source code for Moritz Soelderer's Bachelor Thesis _Should you make use of the ECE metric?
A comparison of Calibration Error Metrics_.

This repository includes a modified version of Scikit-learn, as well as source code of other repositories, 
see License Information below.


## Modifications to Scikit-learn
- Modified the `sklearn.model_selection._search` file (from Scikit-learn) to add the output
    of all used estimators after performing grid search. The modified file can be found
    [here](src/qualitative_analysis/_search_modified.py) `qualitative_analysis._search_modified.py`.
- The original Scikit-learn code is licensed under the BSD 3-Clause License.

## License Information
- The modified Scikit-learn code is licensed under the BSD 3-Clause License.
- The code contributed by Moritz Soelderer is licensed under the MIT License.

See the `LICENSE.md` [here](./LICENSE.md) file for more details.


## Project Structure
- `data_generation` contains code related to the generation of synthetic data and the definitions of synthetic datasets
- `metrics` contains the definitions of calibration error metrics
- `utilities` comprises utility functions such as parameter checks for metrics, as well as utilities for experiments
- `experiments` is home of the experiments introduced in chapter _Experiments_ of the thesis. Each experiment has its own
sub-package containing `run_experiment.py`, `<name_of_experiment>.py` and `<name_of_experiment>_util.py` files and `plots` and 
`logs` sub-directories. `varying_test_sample_size_train_test_split_seeds`, `optimal_ece_assessment.charting_approach` 
as well as all in `misc_scripts` were not discussed in the thesis, as none of these provided insights worth noting.
- `qualitative_analysis` as well was not discussed, and was primarly useful for getting familiar with the metrics.
- `uncertainty_quantification` contains Accuracy Rejection Curves, a concept related to calibration metrics, which finally
did not find any use in this thesis.


## Nomenclature
For the sake of consistency a uniform nomenclature/naming convention is ensured across this 
repository. All source code, except third party code, should adhere to the following rules:

- samples: `X_<additional_args>` (e.g. `X_test`)
- true labels: `y_true_<additional_args>` (e.g. `y_true_test`)
- predicted labels: `y_pred_<additional_args>` (e.g. `y_pred_test`)
- predicted probabilities: `p_pred_<additional_args>` (e.g. `p_pred_test`)
- true probabilities: `p_true_<additional_args>` (e.g. `p_true_test`)