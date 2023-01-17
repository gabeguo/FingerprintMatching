# Inputs: (_01_dist, _02_dist) are distance between anchor and (positive, negative), repsectively
# Returns: (accuracies, fpr, tpr, roc_auc, threshold)
# - accuracies are at every possible threshold
# - fpr is false positive rate at every possible threshold (padded with 0 and 1 at end)
# - tpr is true positive rate at every possible threshold (padded with 0 and 1 at end)
# - roc_auc is scalar: area under fpr (x-axis) vs tpr (y-axis) curve
# - threshold is scalar: below this distance, fingerpritnts match; above, they don't match
def get_metrics(_01_dist, _02_dist):
    all_distances = _01_dist +_02_dist
    all_distances.sort()

    tp, fp, tn, fn = list(), list(), list(), list()
    acc = list()

    # try different thresholds
    for dist in all_distances:
        tp.append(len([x for x in _01_dist if x < dist]))
        tn.append(len([x for x in _02_dist if x >= dist]))
        fn.append(len(_01_dist) - tp[-1])
        fp.append(len(_02_dist) - tn[-1])

        acc.append((tp[-1] + tn[-1]) / len(all_distances))
    # did miss edge case where ideal accuracy threshold is above the maximum distance, 
    # but that's negligible and would only make our results better
    threshold = all_distances[max(range(len(acc)), key=acc.__getitem__)]

    # ROC AUC is FPR = FP / (FP + TN) (x-axis) vs TPR = TP / (TP + FN) (y-axis)
    fpr = [0] + [fp[i] / (fp[i] + tn[i]) for i in range(len(fp))] + [1]
    tpr = [0] + [tp[i] / (tp[i] + fn[i]) for i in range(len(tp))] + [1]
    auc = sum([tpr[i] * (fpr[i] - fpr[i - 1]) for i in range(1, len(tpr))])

    for i in range(1, len(fpr)):
        assert fpr[i] >= fpr[i - 1]
        assert tpr[i] >= tpr[i - 1]

    return acc, fpr, tpr, auc, threshold

_01_dist = [0, 1, 2, 6]
_02_dist = [3, 7, 8, 9]

accuracies, fpr, tpr, roc_auc, threshold = get_metrics(_01_dist, _02_dist)

print('anchor-positive dist:', _01_dist)
print('anchor-negative dist:', _02_dist)
print('thresholds:', sorted(_01_dist + _02_dist))
print('fpr:', fpr)
print('tpr:', tpr)
print('roc auc:', roc_auc)
print('acc:', accuracies)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.grid()
plt.show()

# this checks out