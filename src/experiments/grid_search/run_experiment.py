import os
import sys

from src.experiments.grid_search.grid_search import run
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()

    arguments = sys.argv

    assert len(arguments) == 5, (
        "You should pass 4 arguments: dataset_name, dataset_size, num_folds, true_ece_sample_size"
    )

    dataset_name = arguments[1]
    dataset_size = int(arguments[2])
    num_folds = int(arguments[3])
    true_ece_sample_size = int(arguments[4])

    assert "family" not in dataset_name, "This Experiment is not configured for dataset families"

    # Change working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    run(dataset_name, dataset_size, num_folds, true_ece_sample_size)