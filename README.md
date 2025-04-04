# Project Overview

This repository contains the source code for Moritz Soelderer's Bachelor Thesis.

This repository includes a modified version of Scikit-learn.

## Modifications to Scikit-learn
- Modified the `sklearn.model_selection._search` file (from Scikit-learn) to add the output
    of all used estimators after performing grid search. The modified file can be found
    [here](src/qualitative_analysis/_search_modified.py) `qualitative_analysis._search_modified.py`.
- The original Scikit-learn code is licensed under the BSD 3-Clause License.

## License Information
- The modified Scikit-learn code is licensed under the BSD 3-Clause License.
- The code contributed by Moritz Soelderer is licensed under the MIT License.

See the `LICENSE.md` [here](./LICENSE.md) file for more details.

## Nomenclature
For the sake of consistency a uniform nomenclature/naming convention is ensured across this 
repository. All source code, except third party code, should adhere to the following rules:

- samples: `X_<additional_args>` (e.g. `X_test`)
- true labels: `y_true_<additional_args>` (e.g. `y_true_test`)
- predicted labels: `y_pred_<additional_args>` (e.g. `y_pred_test`)
- predicted probabilities: `p_pred_<additional_args>` (e.g. `p_pred_test`)
- true probabilities: `p_true_<additional_args>` (e.g. `p_true_test`)