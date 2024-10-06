import random

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from metrics.ace import ace
from metrics.balance_score import balance_score
from metrics.ece import ece
from metrics.fce import fce
from metrics.ksce import ksce
from metrics.tce import tce
from metrics.true_ece import true_ece

def plot_pred_prob_dists(steps: np.array, samples: np.ndarray, mode=None):
    print("Plotting Predicted Probability Distributions...")

    if not (mode == "mean" or mode == "variance"):
        raise ValueError("mode not specified. Must be either 'mean' or 'variance'")

    scaling_constant = np.sqrt(2 * np.pi)  # inverse of max. value of gaussian with variance=1
    for step in steps:
        predicted_prob_dist = None
        if mode == "mean":
            predicted_prob_dist = stats.norm(loc=step, scale=1)
        elif mode == "variance":
            predicted_prob_dist = stats.norm(loc=1, scale=step)
        pdf_values = predicted_prob_dist.pdf(samples) * scaling_constant
        # Create the result by stacking the arrays for 1 - p and p
        pred_prob = np.column_stack((1 - pdf_values, pdf_values))
        plt.scatter(samples, pred_prob[:, 1], label=mode + ": " + str(step), s=4)
        plt.xlabel("samples")
        plt.ylabel(mode)
    plt.legend()
    plt.show()


def plot_true_prob_reliability_diagram(steps: np.array, samples: np.ndarray, true_prob: np.ndarray, true_labels: np.array, mode=None):
    print("Plotting True Probability Reliability Diagram...")

    if not (mode == "mean" or mode == "variance"):
        raise ValueError("mode not specified. Must be either 'mean' or 'variance'")

    scaling_constant = np.sqrt(2 * np.pi)  # inverse of max. value of gaussian with variance=1
    for step in steps:
        predicted_prob_dist = None
        if mode == "mean":
            predicted_prob_dist = stats.norm(loc=step, scale=1)
        elif mode == "variance":
            predicted_prob_dist = stats.norm(loc=1, scale=step)
        pdf_values = predicted_prob_dist.pdf(samples) * scaling_constant
        pred_prob = np.column_stack((1 - pdf_values, pdf_values))

        # plotting predicted probabilities against true probabilities
        colors = ['#111111', '#FF5733']  # Define colors for the classes
        class_colors = [colors[label] for label in true_labels]
        plt.scatter(pred_prob[np.arange(pred_prob.shape[0]), true_labels],
                              true_prob[np.arange(true_prob.shape[0]), true_labels], c=class_colors, s=0.9)

        unique_labels = np.unique(true_labels)
        handles = [plt.Line2D([0], [0], marker='o', color='w', label='Class ' + str(label),
                              markerfacecolor=colors[label], markersize=10) for label in unique_labels]

        # Plotting the scatter plot
        plt.xlabel("Predicted Probabilities")
        plt.ylabel("True Probabilities")
        plt.title("Reliability Diagram - " + mode + " " + str(step))
        plt.legend(handles=handles, title="True Labels")
        plt.show()


def plot_metrics(initial, steps, true_prob, true_labels, true_ece_vals, ece_vals, balance_score_vals, fce_vals,
                 tce_vals, ksce_vals, ace_vals, title):
    print("Plotting metrics...")

    # calculating true_prob metrics
    true_prob_ece = ece(true_prob, true_labels, 15)
    true_prob_balance_score = balance_score(true_prob, true_labels)
    true_prob_fce = fce(true_prob, true_labels, 15)
    true_prob_tce = tce(true_prob, true_labels, 0.05, "uniform", 15, 15, 15) * 0.01
    true_prob_ksce = ksce(true_prob, true_labels)
    true_prob_ace = ace(true_prob, true_labels, 15)

    # plotting true ece values
    plt.plot(steps, true_ece_vals, label="true ece values")
    plt.xlabel("Mean")
    plt.ylabel("Metrics")

    # plotting ece values
    plt.plot(steps, ece_vals, label="ece values")
    plt.scatter(initial, true_prob_ece, label="true probability ece")

    # plotting balance score values
    plt.plot(steps, np.abs(balance_score_vals), label="abs. balance score values", linestyle="--")
    plt.scatter(initial, true_prob_balance_score, label="true probability balance score")

    # plotting fce values
    plt.plot(steps, fce_vals, label="fce values")
    plt.scatter(initial, true_prob_fce, label="true probability fce")

    # plotting tce values
    plt.plot(steps, tce_vals, label="normalized tce values", linestyle="--")
    plt.scatter(initial, true_prob_tce, label="true probability normalized tce")

    # plotting ksce values
    plt.plot(steps, ksce_vals, label="ksce values")
    plt.scatter(initial, true_prob_ksce, label="true probability ksce")

    # plotting ace values
    plt.plot(steps, ace_vals, label="ace values")
    plt.scatter(initial, true_prob_ace, label="true probability ace")

    # show plots
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Position outside the top-right corner
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

def main():
    mean = 0
    mean_deviations = np.arange(-3, 4, 0.1)
    true_prob_dist = stats.norm(loc=mean, scale=1)
    samples = true_prob_dist.rvs(size=40000)
    scaling_constant = np.sqrt(2 * np.pi)  # inverse of max. value of gaussian with variance=1

    pdf_values = true_prob_dist.pdf(samples) * scaling_constant
    true_prob = np.column_stack((1 - pdf_values, pdf_values))
    true_labels = np.array(list(map(lambda x: 1 if random.random() < x[1] else 0, true_prob)))

    # plotting some of the conditional probabilities resulting from mean deviation of the underlying distribution (can be deleted later on)
    plot_pred_prob_dists(np.arange(mean - 4, mean + 4, 1), samples, "mean")
    plot_true_prob_reliability_diagram(np.arange(mean - 4, mean + 4, 1), samples, true_prob, true_labels, "mean")

    true_ece_vals = np.array([], dtype=np.float64)
    ece_vals = np.array([], dtype=np.float64)
    balance_score_vals = np.array([], dtype=np.float64)
    fce_vals = np.array([], dtype=np.float64)
    tce_vals = np.array([], dtype=np.float64)
    ksce_vals = np.array([], dtype=np.float64)
    ace_vals = np.array([], dtype=np.float64)

    print("ECE values")
    for mean_deviation in mean_deviations:
        print("mean_deviation", mean_deviation)
        predicted_prob_dist = stats.norm(loc=mean + mean_deviation, scale=1)
        pdf_values = predicted_prob_dist.pdf(samples) * scaling_constant
        pred_prob = np.column_stack((1 - pdf_values, pdf_values))

        # calculating true ece vals
        true_ece_val = true_ece(pred_prob, true_prob)
        true_ece_vals = np.append(true_ece_vals, [true_ece_val])

        # calculating ece vals
        ece_val = ece(pred_prob, true_labels, 15)
        ece_vals = np.append(ece_vals, [ece_val])

        # calculating balance score vals
        balance_score_val = balance_score(pred_prob, true_labels)
        balance_score_vals = np.append(balance_score_vals, [balance_score_val])

        # calculating fce vals
        fce_val = fce(pred_prob, true_labels, 15)
        fce_vals = np.append(fce_vals, [fce_val])

        # calculating tce vals
        tce_val = tce(pred_prob, true_labels, 0.05, "uniform", 15, 15, 15) * 0.01
        tce_vals = np.append(tce_vals, [tce_val])

        # calculating ksce vals
        ksce_val = ksce(pred_prob, true_labels)
        ksce_vals = np.append(ksce_vals, [ksce_val])

        # calculating ace vals
        ace_val = ace(pred_prob, true_labels, 15)
        ace_vals = np.append(ace_vals, [ace_val])

    plot_metrics(mean, mean_deviations, true_prob, true_labels, true_ece_vals, ece_vals, balance_score_vals, fce_vals,
                 tce_vals, ksce_vals, ace_vals, "Metric Behaviour on Mean Deviation")


if __name__ == "__main__":
    main()
