import pickle

import numpy as np
from matplotlib import pyplot as plt

with open('./data/varying_sample_size/Logistic Regression__Iterations_20__AbsoluteValues__20241219_000003.pkl', 'rb') as file:
    results = pickle.load(file)
    print(results)
    means = {
        "True ECE": [],
        "ECE": [],
        "Balance Score": [],
        "FCE": [],
        "KSCE": [],
        "TCE": [],
        "ACE": []
    }

    std_devs = {
        "True ECE": [],
        "ECE": [],
        "Balance Score": [],
        "FCE": [],
        "KSCE": [],
        "TCE": [],
        "ACE": []
    }

    subsample_sizes = []

    # Store Metric Values #
    for result in results:
        for metric, mean in result["means"].items():
            means[metric].append(mean)
        for metric, std in result["std_devs"].items():
            std_devs[metric].append(std)
        subsample_sizes.append(result["Subsample Size"])

    # Plotting Mean and Std Deviation #
    print("   Plotting...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for metric in means.keys():
        if metric != "TCE":
            ax.plot(subsample_sizes, means[metric], label=metric)
            ax.fill_between(subsample_sizes, np.array(means[metric]) - np.array(std_devs[metric]),
                            np.array(means[metric]) + np.array(std_devs[metric]), alpha=0.2)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
    plt.xlabel('Sample Size', fontsize=12)
    plt.ylabel('Metric Values', fontsize=12)
    plt.title('Random Forest__Iterations_20__AbsoluteValues__20241219_052345.pkl', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    #plt.savefig("./plots/varying_sample_size/" + filename + ".png")
    plt.show()