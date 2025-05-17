import os
import sys
from multiprocessing import freeze_support

from src.experiments.varying_bins.varying_bins_dataset_family import run

if __name__ == "__main__":
    freeze_support()

    arguments = sys.argv

    assert len(arguments) == 7, ("You should pass 6 arguments: dataset_name, dataset_size, min_bin_size, max_bin_size, num_steps "
                                 "and true_ece_sample_size")

    dataset_name = arguments[1]
    dataset_size = int(arguments[2])
    min_bin_size = int(arguments[3])
    max_bin_size = int(arguments[4])
    num_steps = int(arguments[5])
    true_ece_sample_size = int(arguments[6])

    assert "family" in dataset_name, ("This Experiment is only configured for dataset families")

    # Change working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    run(dataset_name, dataset_size, min_bin_size, max_bin_size, num_steps, true_ece_sample_size)