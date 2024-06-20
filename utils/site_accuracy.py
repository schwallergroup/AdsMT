import numpy as np
import torch


def get_sites(data, correct=True):
    weights = data.cross_weights
    if correct and 'tags' in data.keys:
        weights[data.tags == 0] = 0.0
        weights = weights / weights.sum()
    sorted, indices = torch.sort(weights, descending=True)
    best_site = [indices[0].item()]
    i = 1
    while i < len(indices) and sorted[0] - sorted[i] < 0.002:
        best_site.append(indices[i].item())
        i += 1
    return best_site, indices


def calc_accuracy(data_list):
    results = []
    for data in data_list:
        if len(data.site) != 0:
            acc = 0
            pred_site, _ = get_sites(data)
            true_site = data.site.tolist()
            for _idx in pred_site:
                if _idx in true_site:
                    acc = 1
            results.append(acc)

    results = np.array(results)
    accuracy = results.sum() / len(results)
    return accuracy
