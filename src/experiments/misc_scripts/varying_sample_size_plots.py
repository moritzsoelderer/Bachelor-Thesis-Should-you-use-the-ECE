import pickle

import numpy as np
from matplotlib import pyplot as plt

filename = 'GummyWorm Dataset__SVM__Iterations_20__AbsoluteValues__20250303_160650'
with (open(f'./data/varying_sample_size/{filename}.pkl', 'rb') as file):
    results = pickle.load(file)

    subsample_sizes = np.linspace(100, 20000, 200, dtype=np.int64)

    # Store Metric Values #

    # Plotting Mean and Std Deviation #
    print("   Plotting...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    means = results["Means"]
    std_devs = results["Std Devs"]

    for metric in means.keys():
        #if metric != "TCE" and metric != "Accuracy" and metric != "True ECE Dists (Binned - 15 Bins)" and metric != "True ECE Dists (Binned - 100 Bins)" and metric != "True ECE Grid (Binned - 15 Bins)" and metric != "True ECE Grid (Binned - 100 Bins)":
            ax.plot(subsample_sizes, means[metric], label=metric)
            #if metric != "True ECE Dists (Binned - 15 Bins)" and metric != "True ECE Dists (Binned - 100 Bins)":
                #ax.fill_between(subsample_sizes, np.array(means[metric]) - np.array(std_devs[metric]),
                           #np.array(means[metric]) + np.array(std_devs[metric]), alpha=0.2)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
    plt.xlabel('Sample Size', fontsize=12)
    plt.ylabel('Metric Values', fontsize=12)
    plt.title(filename, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.legend()
    plt.ylim(0.003, 0.012)
    ax.grid(True, linestyle='--', alpha=0.6)

    #plt.savefig("./plots/varying_sample_size/" + filename + ".png")
    plt.show(block=False)

    print("Std Dev 15 Bins", results["Std Devs"]["True ECE Dists (Binned - 15 Bins)"][0])
    print("Std Dev 100 Bins", results["Std Devs"]["True ECE Dists (Binned - 100 Bins)"][0])