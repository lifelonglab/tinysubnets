import math
import torch


def asymmetric(input_value, sf, bits, mean_val=0, scale_val=1):
    """

    :param input_value:
    :param sf:
    :param bits:
    :param mean_val:
    :param scale_val:
    :return:
    """
    assert bits >= 1, bits

    if bits == 1:
        return torch.sign(input_value) - 1

    delta = math.pow(2.0, -sf) * scale_val
    bound = math.pow(2.0, bits - 1) / scale_val
    min_val = - bound
    max_val = bound - 1
    input_value = torch.add(input_value, -mean_val)
    rounded = torch.floor(input_value / delta + 0.5)
    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    clipped_value = torch.add(clipped_value, mean_val)

    return clipped_value
