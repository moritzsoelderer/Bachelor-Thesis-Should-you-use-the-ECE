import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy import stats

from src.utilities import data_generation as dg
from src.metrics.ace import ace
from src.metrics.balance_score import balance_score
from src.metrics.ece import ece
from src.metrics.fce import fce
from src.metrics.ksce import ksce
from src.metrics.tce import tce
from src.metrics.true_ece import true_ece


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


def plot_true_prob_reliability_diagrams(steps: np.array, samples: np.ndarray, true_prob: np.ndarray, true_labels: np.array, calcProbs: callable, title: str = ""):
    print("Plotting True Probability Reliability Diagram...")

    for step in steps:
        pred_prob = calcProbs(samples, step)

        # plotting predicted probabilities against true probabilities
        colors = ['#111111', '#FF5733']
        class_colors = [colors[label] for label in true_labels]
        plt.scatter(true_prob[:, 1], pred_prob[:, 1], c=class_colors, s=0.9)

        unique_labels = np.unique(true_labels)
        handles = [plt.Line2D([0], [0], marker='o', color='w', label='Class ' + str(label),
                              markerfacecolor=colors[label], markersize=10) for label in unique_labels]
        # Plotting the scatter plot
        plt.ylabel("Predicted Probabilities")
        plt.xlabel("True Probabilities")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Reliability Diagram - " + title + " " + str(step))
        plt.legend(handles=handles, title="True Labels")
        plt.show()


def plot_true_prob_reliability_diagram(true_prob, pred_prob, true_labels, title="Reliability Diagram"):
    colors = ['#111111', '#FF5733']
    class_colors = [colors[label] for label in true_labels]
    plt.scatter(true_prob[:, 1], pred_prob[:, 1], c=class_colors, s=0.9)

    unique_labels = np.unique(true_labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label='Class ' + str(label),
                          markerfacecolor=colors[label], markersize=10) for label in unique_labels]
    # Plotting the scatter plot
    plt.ylabel("Predicted Probabilities")
    plt.xlabel("True Probabilities")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)
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

        print("True ECE: ", true_ece_val)
        print("ECE: ", ece_val)
        print("Balance Score: ", balance_score_val)
        print("FCE: ", fce_val)
        print("TCE: ", tce_val)
        print("KSCE: ", ksce_val)
        print("ACE: ", ace_val)

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


def svm1_probs(samples, step):
    prob_dist = stats.norm(loc=0, scale=1)
    noise_dist = stats.norm(loc=0, scale=step)
    pdf_values = prob_dist.pdf(samples) * np.sqrt(2 * np.pi)
    noise = noise_dist.rvs(size=samples.shape[0]) * ((pdf_values + 0.1) * (1 - pdf_values + 0.1)) ** 1.5 * 10
    noisy_pdf_values = pdf_values
    noisy_pdf_values = np.clip(noisy_pdf_values, 0, 1)
    sine_transformatted_pdf_values = (1.0 / 2.0 ** 1.1) * (np.sin(noisy_pdf_values * np.pi - 1.5) + 1) ** 1.1
    sine_transformatted_pdf_values = sine_transformatted_pdf_values * 0.8 + 0.1 + noise
    if step == 0:
        sine_transformatted_pdf_values = noisy_pdf_values
    prob = np.column_stack((1 - sine_transformatted_pdf_values, sine_transformatted_pdf_values))
    print("avg noise ", np.average(np.abs(noise)))
    return prob


def svm2_probs(samples, step):
    prob_dist = stats.norm(loc=0, scale=1)
    noise_dist = stats.norm(loc=0, scale=step)
    pdf_values = prob_dist.pdf(samples) * np.sqrt(2 * np.pi)
    noise = noise_dist.rvs(size=samples.shape[0]) * ((pdf_values + 0.1) * (1 - pdf_values + 0.1)) ** 1.5 * 10
    noisy_pdf_values = pdf_values
    noisy_pdf_values = np.clip(noisy_pdf_values, 0, 1)
    sine_transformatted_pdf_values =  (1.0 / 2.0 ** 1.1) * (np.sin(noisy_pdf_values * np.pi - 1.5) + 1) ** 1.1
    sine_transformatted_pdf_values = sine_transformatted_pdf_values * 0.8 + 0.1 + noise
    if step == 0:
        sine_transformatted_pdf_values = noisy_pdf_values
    sine_transformatted_pdf_values = np.clip(sine_transformatted_pdf_values, 0, 1.25) * 0.8
    prob = np.column_stack((1 - sine_transformatted_pdf_values, sine_transformatted_pdf_values))
    print("avg noise ", np.average(np.abs(noise)))
    return prob


def randomforest1_probs(samples, step):
    prob_dist = stats.norm(loc=0, scale=1)
    noise_dist = stats.norm(loc=0, scale=step)

    pdf_values = prob_dist.pdf(samples) * np.sqrt(2 * np.pi)
    noise = noise_dist.rvs(size=samples.shape[0]) * ((pdf_values + 0.05) * (1 - pdf_values + 0.05)) ** 1.5 * 10

    noisy_pdf_values = pdf_values + noise
    noisy_pdf_values = np.clip(noisy_pdf_values, 0, 1)

    prob = np.column_stack((1 - noisy_pdf_values, noisy_pdf_values))
    print("avg noise ", np.average(np.abs(noise)))
    return prob


def randomforest2_probs(samples, step):
    prob_dist = stats.norm(loc=0, scale=1)
    noise_dist = stats.norm(loc=0, scale=step)

    pdf_values = prob_dist.pdf(samples) * np.sqrt(2 * np.pi)
    noise = noise_dist.rvs(size=samples.shape[0]) * ((pdf_values + 0.05) * (1 - pdf_values + 0.05)) ** 1.5 * 10

    noisy_pdf_values = pdf_values + noise
    noisy_pdf_values = np.clip(noisy_pdf_values, 0, 1.25) * 0.8

    prob = np.column_stack((1 - noisy_pdf_values, noisy_pdf_values))
    print("avg noise ", np.average(np.abs(noise)))
    return prob


def logisticregression1_probs(samples, step):
    prob_dist = stats.norm(loc=0, scale=1)
    noise_dist = stats.norm(loc=0, scale=step)

    pdf_values = prob_dist.pdf(samples) * np.sqrt(2 * np.pi)
    noise = noise_dist.rvs(size=samples.shape[0]) * 4 * np.abs(pdf_values - 0.5)

    noisy_pdf_values = pdf_values + noise
    noisy_pdf_values = np.clip(noisy_pdf_values, 0, 1)

    prob = np.column_stack((1 - noisy_pdf_values, noisy_pdf_values))
    print("avg noise ", np.average(np.abs(noise)))
    return prob


def logisticregression2_probs(samples, step):
    prob_dist = stats.norm(loc=0, scale=1)
    noise_dist = stats.norm(loc=0, scale=step)

    pdf_values = prob_dist.pdf(samples) * np.sqrt(2 * np.pi)
    noise = np.abs(noise_dist.rvs(size=samples.shape[0])) * ((pdf_values + 0.1) * (1 - pdf_values + 0.1)) ** 1.5 * 10

    noisy_pdf_values = pdf_values
    noisy_pdf_values = np.clip(noisy_pdf_values, 0, 1)

    polynomial_transformatted_pdf_values = 2 * (1.1 * noisy_pdf_values - 0.36) ** 3 + 0.2
    polynomial_transformatted_pdf_values = polynomial_transformatted_pdf_values + noise
    if step == 0:
        polynomial_transformatted_pdf_values = noisy_pdf_values
    sigmoid_transformatted_pdf_values = np.clip(polynomial_transformatted_pdf_values, 0, 1) * 0.8
    prob = np.column_stack((1 - sigmoid_transformatted_pdf_values, sigmoid_transformatted_pdf_values))
    print("avg noise ", np.average(np.abs(noise)))
    return prob

def svm_interpolated_probs(samples, step):
    prob_dist = stats.norm(loc=0, scale=1)
    noise_dist = stats.norm(loc=0, scale=0.015)
    pdf_values = prob_dist.pdf(samples) * np.sqrt(2 * np.pi)

    transformatted_pdf_values = 1 / (1 + np.exp(- step * (pdf_values - 0.5)))

    noise = noise_dist.rvs(size=samples.shape[0]) * ((transformatted_pdf_values + 0.1) * (1 - transformatted_pdf_values + 0.1)) ** 1.5 * 10
    noisy_pdf_values = transformatted_pdf_values + noise
    noisy_pdf_values = np.clip(noisy_pdf_values, 0, 1)

    if step == 3:
        noisy_pdf_values = np.clip(pdf_values + noise, 0, 1)

    prob = np.column_stack((1 - noisy_pdf_values, noisy_pdf_values))
    print("avg noise ", np.average(np.abs(noise)))
    return prob