from torch.autograd import Variable
import torch
import numpy as np
import math
from ..basic_routines.re_fexp import re_fexp, re_fexp_cuda, re_fexp_hist, re_fexp_hist_cuda
from ..basic_routines.asymmetric import asymmetric


def compute_integral_part(data_in):
    """
    Used for linear quantization.
    :param data_in:
    :return:
    """
    if isinstance(data_in, Variable):
        data_in = data_in.data.cpu().numpy()
    elif torch.is_tensor(data_in):
        data_in = data_in.numpy()
    max_coeff_abs = np.max(np.abs(data_in))
    int_bits = int(math.ceil(math.log(max_coeff_abs, 2)))
    return int_bits


def integral_part_for_asymmetric(input_value, mean_val=0):
    """
    Used for asymmetric quantization.
    :param input_value:
    :param mean_val:
    :return:
    """
    data_in = input_value - mean_val
    if isinstance(data_in, Variable):
        data_in = data_in.data.cpu().numpy()
    elif torch.is_tensor(data_in):
        data_in = data_in.numpy()
    max_coeff_abs = np.max(np.abs(data_in))
    sf = math.ceil(math.log2(max_coeff_abs + 1e-12))
    return sf

# def integral_part_for_asymmetric(input_value, mean_val=0):
#
#     input2 = input_value - mean_val
#
#     abs_value = np.absolute(input2)
#
#     v = abs_value
#     print("v shape: ", v.shape)
#
#     if isinstance(v, Variable):
#
#         v = v.data.cpu().numpy()[0]
#         print("v shape: ", v.shape)
#
#     sf = math.ceil(math.log2(v+1e-12))
#
#     return sf


def mean_for_asymmetric(data_in):
    """

    :param data_in:
    :return:
    """
    return (data_in.max() + data_in.min())/2


def scale_for_asymmetric(data_in, integer_bits):
    """

    :param data_in:
    :param integer_bits:
    :return:
    """
    return data_in.abs().max()/math.pow(2.0, integer_bits)


# to make it more explicit
def linear_(data_in, frac_len, word_len):
    if isinstance(data_in, Variable):
        if data_in.data.is_cuda:
            data_out = re_fexp_cuda(data_in=data_in, frac_len=frac_len, word_len=word_len)
        else:
            data_in = data_in.data.cpu().numpy()
            data_out = re_fexp(data_in=data_in, frac_len=frac_len, word_len=word_len)
            data_out = Variable(torch.from_numpy(data_out))

    elif torch.is_tensor(data_in):
        data_in = data_in.cpu().numpy()
        data_out = re_fexp(data_in=data_in, frac_len=frac_len, word_len=word_len)
        data_out = torch.from_numpy(data_out)
    return data_out


# to make it more explicit
def linear_hist(data_in, frac_len, word_len):
    if isinstance(data_in, Variable):
        if data_in.data.is_cuda:
            data_out = re_fexp_hist_cuda(data_in=data_in, frac_len=frac_len, word_len=word_len)
        else:
            data_in = data_in.data.cpu().numpy()
            data_out = re_fexp_hist(data_in=data_in, frac_len=frac_len, word_len=word_len)
            data_out = Variable(torch.from_numpy(data_out))

    elif torch.is_tensor(data_in):
        data_in = data_in.cpu().numpy()
        data_out = re_fexp_hist(data_in=data_in, frac_len=frac_len, word_len=word_len)
        data_out = torch.from_numpy(data_out)
    return data_out


# to make it more explicit
def asymmetric_(input_value, sf, bits, mean_val=0, scale_val=1):
    """
    Only Torch procedure is available (no Numpy implementation).
    :param input_value:
    :param sf:
    :param bits:
    :param mean_val:
    :param scale_val:
    :return:
    """
    data_out = asymmetric(input_value=input_value,
                          sf=sf,
                          bits=bits,
                          mean_val=mean_val,
                          scale_val=scale_val)
    return data_out