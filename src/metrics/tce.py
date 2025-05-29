import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binomtest, t


"""
    This code was adapted from:
    
    **TCE: A Test-Based Approach to Measuring Calibration Error**  
    GitHub Repository: https://github.com/facebookresearch/tce
    
    Changes were made to lines 17, 18, 19, 39, 40, 41  
"""

def tce(preds, labels, siglevel=0.05, strategy='pavabc', n_min=10, n_max=1000, n_bin=10, savepath=False, ymax=None):
    assert labels.shape[0] != n_min, "The minimum bin size equals to the data size. No binning needed."

    ### Added by myself so that I am not obligated to use a 1d array with positive class labels
    preds = np.array([elem[1] for elem in preds], dtype=np.float32)
    ###

    bin_preds, bin_count, bin_total, _ = calibration_summary(preds, labels, strategy, n_min=n_min, n_max=n_max, n_bin=n_bin)

    bin_rnum = np.zeros(len(bin_count))
    for i in range(len(bin_rnum)):
        pvals = np.array([ binomtest(bin_count[i], bin_total[i], p=p).pvalue for p in bin_preds[i] ])
        bin_rnum[i] = sum((pvals <= siglevel))

    if savepath != False:
        plot_tce_diagram(bin_rnum, bin_preds, bin_count, bin_total, savepath, ymax)

    return 100 * np.sum(bin_rnum) / np.sum(bin_total)
    # return np.sum(bin_rnum) / np.sum(bin_total), bin_rnum, bin_preds, bin_count, bin_total



def tce_ttest(preds, labels, siglevel=0.05, strategy='pavabc', n_min=10, n_max=1000, n_bin=10, savepath=False, ymax=None):
    assert labels.shape[0] != n_min, "The minimum bin size equals to the data size. No binning needed."

    ### Added by myself so that I am not obligated to use a 1d array with positive class labels
    preds = np.array([elem[1] for elem in preds], dtype=np.float32)
    ###

    bin_preds, bin_count, bin_total, _ = calibration_summary(preds, labels, strategy, n_min=n_min, n_max=n_max, n_bin=n_bin)

    bin_rnum = np.zeros(len(bin_count))
    for i in range(len(bin_rnum)):
        ni = bin_total[i]
        mu = bin_count[i] / ni
        sd = mu * ( 1 - mu )
        if sd == 0:
            bin_rnum[i] = len(bin_preds[i])
        else:
            pvals = np.array([ 2.0 *t.sf(np.sqrt(ni) *np.abs(mu -p ) /sd, ni -1) for p in bin_preds[i] ])
            bin_rnum[i] = sum((pvals <= siglevel))

    if savepath != False:
        plot_tce_diagram(bin_rnum, bin_preds, bin_count, bin_total, savepath, ymax)

    return 100 * np.sum(bin_rnum) / np.sum(bin_total)
    # return np.sum(bin_rnum) / np.sum(bin_total), bin_rnum, bin_preds, bin_count, bin_total


def calibration_summary(preds, labels, strategy='pavabc', n_min=10, n_max=1000, n_bin=10):
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0), "Prediction Out of Range [0, 1]"
    assert np.all((labels == 0) | (labels == 1)), "Label Not 0 or 1"

    if strategy == 'pavabc':
        bin_preds, bin_count, bin_total, bins = _pavabc(preds, labels, n_min=n_min, n_max=n_max)
    elif strategy == 'pava':
        bin_preds, bin_count, bin_total, bins = _pavabc(preds, labels, n_min=0, n_max=len(preds) + 1)
    elif strategy == 'uniform':
        bin_preds, bin_count, bin_total, bins = _calibration_process(preds, labels, strategy, n_bin)
    elif strategy == 'quantile':
        bin_preds, bin_count, bin_total, bins = _calibration_process(preds, labels, strategy, n_bin)
    else:
        assert False, 'no correct strategy specified: (uniform, quantile, pava, ncpave)'

    return bin_preds, bin_count, bin_total, bins


def _pavabc(x, y, n_min=0, n_max=10000):
    ### Sort (Start) ###
    order = np.argsort(x)
    xsort = x[order]
    ysort = y[order]
    num_y = len(ysort)

    ### Sort (End) ###

    def _condition(y0, y1, w0, w1):
        condition1 = (w0 + w1 <= n_min)
        condition2 = (w0 + w1 <= n_max)
        condition3 = (y0 / w0 >= y1 / w1)
        return condition1 or (condition2 and condition3)

    ### PAVA with Number Constraint (Start) ###
    count = -1
    iso_y = []
    iso_w = []
    for i in range(num_y - n_min):
        count += 1
        iso_y.append(ysort[i])
        iso_w.append(1)
        while count > 0 and _condition(iso_y[count - 1], iso_y[count], iso_w[count - 1], iso_w[count]):
            iso_y[count - 1] += iso_y[count]
            iso_w[count - 1] += iso_w[count]
            iso_y.pop()
            iso_w.pop()
            count -= 1
    if n_min > 0:
        count += 1
        iso_y.append(sum(ysort[num_y - n_min:num_y]))
        iso_w.append(n_min)
        if iso_w[-1] + n_min <= n_max:
            iso_y[count - 1] += iso_y[count]
            iso_w[count - 1] += iso_w[count]
            iso_y.pop()
            iso_w.pop()
            count -= 1
    ### PAVA with Number Constraint (End) ###

    ### Process return values (Start) ###
    index = np.r_[0, np.cumsum(iso_w)]
    bins = np.r_[0.0, [(xsort[index[j] - 1] + xsort[index[j]]) / 2.0 for j in range(1, len(index) - 1)], 1.0]
    bin_count = np.array(iso_y)
    bin_total = np.array(iso_w)
    bin_preds = [xsort[index[j]:index[j + 1]] for j in range(len(index) - 1)]
    ### Process return values (End) ###

    return bin_preds, bin_count, bin_total, bins


def _calibration_process(preds, labels, strategy="uniform", n_bin=10):
    if strategy == 'uniform':
        bins = np.linspace(0.0, 1.0, n_bin + 1)
        bins[-1] = 1.1  # trick to include 'pred=1.0' in the final bin
        indices = np.digitize(preds, bins, right=False) - 1
        bins[-1] = 1.0  # put it back to 1.0
        bin_count = np.array([sum(labels[indices == i]) for i in range(bins.shape[0] - 1)]).astype(int)
        bin_total = np.array([len(labels[indices == i]) for i in range(bins.shape[0] - 1)]).astype(int)
        bin_preds = [preds[indices == i] for i in range(bins.shape[0] - 1)]
        return bin_preds, bin_count, bin_total, bins

    elif strategy == 'quantile':
        quantile = np.linspace(0, 1, n_bin + 1)
        # bins = np.percentile(preds, quantile * 100)
        # bins[0] = 0.0
        # bins[-1] = 1.0
        sortedindices = np.argsort(preds)
        sortedlabels = labels[sortedindices]
        sortedpreds = preds[sortedindices]
        idpartition = (quantile * len(labels)).astype(int)
        bin_count = np.array([sum(sortedlabels[s:e]) for s, e in zip(idpartition, idpartition[1:])]).astype(int)
        bin_total = np.array([len(sortedlabels[s:e]) for s, e in zip(idpartition, idpartition[1:])]).astype(int)
        bin_preds = [sortedpreds[s:e] for s, e in zip(idpartition, idpartition[1:])]
        bins = np.array([0.0] + [(sortedpreds[e - 1] + sortedpreds[e]) / 2.0 for e in idpartition[1:-1]] + [1.0])
        return bin_preds, bin_count, bin_total, bins

    else:
        assert False, 'no correct strategy specified: (uniform, quantile)'


def plot_tce_diagram(bin_rnum, bin_preds, bin_count, bin_total, savepath=False, ymax=None):
    ### Prepare values (start) ###
    bin_prob = np.zeros(len(bin_total))
    bin_prob[bin_total != 0] = bin_count[bin_total != 0] / bin_total[bin_total != 0]

    width = 1 / (bin_total.shape[0] + 1)
    positions = np.linspace(0.0, 1.0, bin_total.shape[0] + 1)[:-1] + width / 2.0
    if ymax == None:
        ymax = np.maximum(max(bin_prob), np.array([elem.mean() for elem in bin_preds]).max())
        ymax = 1.25 * ymax if ymax < 0.7 else 1.0
    ### Prepare values (end) ###

    ### Plot (start) ###
    ratio = 4
    fig, axs = plt.subplots(2, 2, figsize=(6, 6), gridspec_kw={'width_ratios': [ratio, 1], 'height_ratios': [ratio, 1]})

    axs[1, 1].set_visible(False)

    axs[0, 1].hist(np.concatenate(bin_preds), bins=30, orientation="horizontal")
    axs[0, 1].set_box_aspect(ratio / 1)
    axs[0, 1].set_ylim(0, ymax)
    axs[0, 1].set_yticklabels([])
    axs[0, 1].xaxis.set_label_position("top")
    axs[0, 1].xaxis.tick_top()
    axs[0, 1].tick_params(axis='x', labelsize=12)
    axs[0, 1].set_xlabel("Count", fontsize=14)

    axs[1, 0].bar(positions, bin_total, width=width, color="grey", alpha=0.5, linewidth=3)
    axs[1, 0].bar(positions, bin_rnum, width=width, color="red", alpha=0.5, linewidth=3)
    axs[1, 0].set_box_aspect(1 / ratio)
    axs[1, 0].set_xlim(0, 1.0)
    axs[1, 0].set_xticks(positions)
    axs[1, 0].set_xticklabels(["{:d}".format(i + 1) for i in range(positions.shape[0])])
    axs[1, 0].tick_params(axis='x', labelsize=12)
    axs[1, 0].tick_params(axis='y', labelsize=12)
    axs[1, 0].set_xlabel("Bin ID", fontsize=14)
    axs[1, 0].set_ylabel("Count", fontsize=14)

    conf_plt = axs[0, 0].violinplot(bin_preds, positions, widths=width * 0.8, vert=True, showmeans=True,
                                    showextrema=True, showmedians=False, bw_method=None)
    accr_plt = axs[0, 0].hlines(bin_prob, positions - (0.8 * width / 2.0), positions + (0.8 * width / 2.0),
                                linestyle="-", linewidth=3, color="red", label="Empirical Probability")
    axs[0, 0].set_box_aspect(1)
    axs[0, 0].set_xlim(0, 1.0)
    axs[0, 0].set_ylim(0, ymax)
    axs[0, 0].set_xticks(positions)
    axs[0, 0].set_xticklabels([])
    axs[0, 0].xaxis.set_label_position("top")
    axs[0, 0].set_xlabel(" ", fontsize=14)
    axs[0, 0].set_ylabel(r"$P_\theta(y=1 \mid x)$", fontsize=14)
    axs[0, 0].set_title(r"Estimates vs Predictions", fontsize=14)
    axs[0, 0].tick_params(axis='y', labelsize=12)
    axs[0, 0].legend(handles=[accr_plt], loc='upper left', fontsize=12)
    ### Plot (end) ###

    ### Save (start) ###
    if not savepath == False:
        fig.savefig(savepath, dpi=288)
        plt.close(fig)
    ### Plot (end) ###