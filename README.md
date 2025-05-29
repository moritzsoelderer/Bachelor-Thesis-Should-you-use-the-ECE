## Project Overview


### General Information
This repository contains the source code for Moritz Soelderer's Bachelor Thesis _Should you make use of the ECE metric?
A comparison of Calibration Error Metrics_.

The thesis gives insights regarding the reliability of the Expected Calibration Error (ECE) and compares it thereby,
along with other calibration error metrics, to a binned approximation of the True Expected Calibration Error or True Calibration Error.

For detailed information about conducted experiments and findings, please refer to the thesis itself.




### License
- The code contributed by Moritz Soelderer is licensed under the MIT License.
- Third party code and libraries are licensed under their respective license.

See the `LICENSE.md` [here](./LICENSE.md) file for more details.

### Modifications to Scikit-learn
This codebase contains a modified code from the scikit-learn library. 
The `GridSearchCV` class located in `sklearn.model_selection._search`, was modified in a way
that it outputs all trained estimators after performing grid search. The modified file can be found
[here](src/experiments/grid_search/_search_modified.py).

### Project Structure
- `data_generation` contains code related to the generation of synthetic data and the definitions of synthetic datasets
- `metrics` contains the definitions of calibration error metrics
- `utilities` comprises utility functions such as parameter checks for metrics, as well as utilities for experiments
- `experiments` is home of the experiments introduced in chapter _Experiments_ of the thesis. Each experiment has its own
sub-package containing `run_experiment.py`, `<name_of_experiment>.py` and `<name_of_experiment>_util.py` files and `plots` and 
`logs` sub-directories. `varying_test_sample_size_train_test_split_seeds`, `optimal_ece_assessment` 
as well as `misc_scripts` were not discussed in the thesis, as none of these provided insights worth noting.
- `qualitative_analysis` as well was not discussed, and was primarly useful for getting familiar with the metrics.
- `uncertainty_quantification` contains Accuracy Rejection Curves, a concept related to calibration metrics, which finally
did not find any use in this thesis.


### Nomenclature
For the sake of consistency a uniform nomenclature/naming convention is mostly ensured across this 
repository. All source code, except third party code, should adhere to the following rules:

- samples: `X_<additional_args>` (e.g. `X_test`)
- true labels: `y_true_<additional_args>` (e.g. `y_true_test`)
- predicted labels: `y_pred_<additional_args>` (e.g. `y_pred_test`)
- predicted probabilities: `p_pred_<additional_args>` (e.g. `p_pred_test`)
- true probabilities: `p_true_<additional_args>` (e.g. `p_true_test`)