import numpy as np
import math


def channel_to_recursive_adaptive_grid(data_channel, grid):
    sorted_channel = np.sort(data_channel)

    def find_edges(data, total_needed, found_edges):
        total = data.shape[0]

        if total == 0:  # there is no more unique data left
            return found_edges

        still_needed = total_needed - len(found_edges)
        candidate_edges = [
                              data[int(math.floor((total * (i + 1.0)) / (still_needed + 1)))]
                              for i in range(still_needed)
                          ] + found_edges

        unique_edges = list(set(candidate_edges))

        if len(unique_edges) == total_needed:
            return unique_edges

        duplicated_edges = {e for e in unique_edges if candidate_edges.count(e) > 1}
        found_edges = list(set(found_edges + list(duplicated_edges)))

        return find_edges(np.compress([
            n not in duplicated_edges for n in data
        ], data), total_needed, found_edges)

    edges = find_edges(sorted_channel[1:-1], grid + 1, [sorted_channel[0], sorted_channel[-1]])
    return np.sort(edges)


def hist_quant_adaptive(data_in, word_len=2, method='means'):
    grid = pow(2, word_len)
    edges = channel_to_recursive_adaptive_grid(data_channel=data_in.flatten(), grid=grid)
    inds = np.digitize(data_in, edges[1:grid])

    means = []
    medians = []
    for g in range(grid):
        condition = (inds == g).reshape((-1,))
        values = np.compress(condition, data_in)
        means.append(np.mean(values))
        medians.append(np.median(values))

    if method == 'medians':
        data_out = np.array(medians)[inds]
    elif method == 'edges':
        data_out = edges[inds]
    else:  # method == 'means':
        data_out = np.array(means)[inds]

    print("Coeff. quant adapt. difference sum: ", np.sum((data_in - data_out)))
    print("Coeff. quant adapt. difference mean: ", np.mean((data_in - data_out)))

    return data_out


def hist_quant(data_in, word_len=2):
    """
    Simple histogram quantization.
    :param data_in:
    :param word_len: no. of bits.
    :return:
    """

    print("doing hist_quant.")
    # get histogram
    h = np.histogram(data_in.flatten(), normed=False, bins=pow(2, word_len))

    # pick middle of the edges
    #bins = [(h[1][i] + h[1][i + 1])/2 for i in range(h[1].size - 1)]
    bins = [h[1][i] for i in range(h[1].size - 1)]

    # account for the last edge (copy it)
    bins.append(bins[-1])

    # convert to the dict structure for the performance sake
    bins_dict = {i: bins[i] for i in range(len(bins))}

    # map input data to indices of h[1] - edges of the histogram (h[0]- are hist values)
    inds = np.digitize(data_in, h[1], right=False) - 1

    # map the indices to the values (the average of a bucket edges)
    data_out = h[1][inds]

    return data_out


def hist_quant_cuda(data_in, word_len=2):
    pass




