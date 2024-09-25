# This file contains my implementations of the standard methods
from ..basic_routines.hist_quant import hist_quant, hist_quant_cuda, hist_quant_adaptive
from ..basic_routines.clustering import cluster_kmeans, cluster_kmeans_cuda
from torch.autograd import Variable
import torch


# to make it more explicit
def wrapped_hist_quant_adaptive(data_in, word_len):
    if isinstance(data_in, Variable):
        if data_in.data.is_cuda:
            data_out = hist_quant_cuda(data_in=data_in, word_len=word_len)
        else:
            data_in = data_in.data.cpu().numpy()
            data_out = hist_quant_adaptive(data_in=data_in, word_len=word_len)
            data_out = Variable(torch.from_numpy(data_out))

    elif torch.is_tensor(data_in):
        data_in = data_in.cpu().numpy()
        data_out = hist_quant_adaptive(data_in=data_in, word_len=word_len)
        data_out = torch.from_numpy(data_out)
    return data_out


# kmeans clustering
def wrapped_cluster_kmeans(data_in, word_len):
    if isinstance(data_in, Variable):
        if data_in.data.is_cuda:
            data_out = cluster_kmeans_cuda(data_in=data_in, word_len=word_len)
        else:
            data_in = data_in.data.cpu().numpy()
            data_out = cluster_kmeans(data_in=data_in, word_len=word_len)
            data_out = Variable(torch.from_numpy(data_out))

    elif torch.is_tensor(data_in):
        data_in = data_in.cpu().numpy()
        data_out = cluster_kmeans(data_in=data_in, word_len=word_len)
        data_out = torch.from_numpy(data_out)
        return data_out
