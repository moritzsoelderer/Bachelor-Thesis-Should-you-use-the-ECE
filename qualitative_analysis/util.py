import numpy as np
import matplotlib.pyplot as plt

from metrics.ace import ace
from metrics.balance_score import balance_score
from metrics.ece import ece
from metrics.fce import fce
from metrics.ksce import ksce
from metrics.tce import tce
from metrics.true_ece import true_ece


def plot_pred_prob_dists(steps: np.array, samples: np.ndarray, calcProbs: callable, label: str, title="Predicted Probability Distributions"):
    print("Plotting Predicted Probability Distributions...")

    for step in steps:
        pred_prob = calcProbs(samples, step)
        plt.scatter(samples, pred_prob[:, 1], label=label + ": " + str(step), s=4)
        plt.xlabel("Samples")
        plt.ylabel("Predicted Probability")
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_true_prob_reliability_diagram(steps: np.array, samples: np.ndarray, true_prob: np.ndarray, true_labels: np.array, calcProbs: callable,title: str = ""):
    print("Plotting True Probability Reliability Diagram...")

    for step in steps:
        pred_prob = calcProbs(samples, step)

        # plotting predicted probabilities against true probabilities
        colors = ['#111111', '#FF5733']
        class_colors = [colors[label] for label in true_labels]
        plt.scatter(pred_prob[:, 1], true_prob[:, 1], c=class_colors, s=0.9)

        unique_labels = np.unique(true_labels)
        handles = [plt.Line2D([0], [0], marker='o', color='w', label='Class ' + str(label),
                              markerfacecolor=colors[label], markersize=10) for label in unique_labels]
        # Plotting the scatter plot
        plt.xlabel("Predicted Probabilities")
        plt.ylabel("True Probabilities")
        plt.title("Reliability Diagram - " + title + " " + str(step))
        plt.legend(handles=handles, title="True Labels")
        plt.show()

def calculate_metrics(steps, true_prob, true_labels, samples, calcProbs, log="Step"):
    true_ece_vals = np.array([], dtype=np.float64)
    ece_vals = np.array([], dtype=np.float64)
    balance_score_vals = np.array([], dtype=np.float64)
    fce_vals = np.array([], dtype=np.float64)
    tce_vals = np.array([], dtype=np.float64)
    ksce_vals = np.array([], dtype=np.float64)
    ace_vals = np.array([], dtype=np.float64)

    for step in steps:
        print(log + ": ", step)
        pred_prob = calcProbs(samples, step)

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

    return true_ece_vals, ece_vals, balance_score_vals, fce_vals, tce_vals, ksce_vals, ace_vals

def plot_metrics(initial, steps, true_prob, true_labels, true_ece_vals, ece_vals, balance_score_vals, fce_vals,
                 tce_vals, ksce_vals, ace_vals, title, xlabel):
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
    plt.xlabel(xlabel)
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
