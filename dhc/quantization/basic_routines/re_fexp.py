import numpy as np
import torch

clip_table = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
              131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864,
              134217728, 268435456, 536870912, 1.0737418240e+009, 2.1474836480e+009, 4.2949672960e+009,
              8.589934592000000e+09, 1.717986918400000e+10, 3.435973836800000e+10, 6.871947673600000e+10,
              1.374389534720000e+11, 2.748779069440000e+11, 5.497558138880000e+11, 1.099511627776000e+12,
              2.199023255552000e+12, 4.398046511104000e+12, 8.796093022208000e+12, 1.759218604441600e+13,
              3.518437208883200e+13, 7.036874417766400e+13, 1.407374883553280e+14, 2.814749767106560e+14,
              5.629499534213120e+14, 1.125899906842624e+15, 2.251799813685248e+15, 4.503599627370496e+15,
              9.007199254740992e+15, 1.801439850948198e+16, 3.602879701896397e+16, 7.205759403792794e+16,
              1.441151880758559e+17, 2.882303761517117e+17, 5.764607523034235e+17, 1.152921504606847e+18]

clip_table = np.array(clip_table)


if torch.cuda.is_available():
    cuda_clip_table = torch.from_numpy(clip_table).float().cuda()


def check_ok(cuda, reg):

    cuda = cuda.cpu().numpy()

    #print("cuda: ", cuda)
    #print("reg: ", reg)

    if (cuda == reg).all():
        print('ok')
    else:
        print('wrong')
        print(np.mean(np.abs(cuda - reg)))


def re_fexp(data_in, word_len, frac_len):
    """
    Pure Python implementation of the fixed-point casting.
    :type frac_len: int
    :param data_in:
    :param frac_len: bits
    :param enable:
    :return:
    """

    if frac_len < 0:
        scale_factor = 1/clip_table[-frac_len]
    else:
        scale_factor = clip_table[frac_len]

    scale_factor_inv = 1/scale_factor

    data_out = np.round(data_in*scale_factor)

    # Sauration limit
    sat_limit = clip_table[word_len]  #  i.e. clip_table((word_len-1) + 1);

    #print(data_out.shape)
    #print(sat_limit)

    np.clip(data_out, (1 - sat_limit), (sat_limit - 1), data_out)
    # print "python clipped: ", data_out

    data_out *= scale_factor_inv

    # print('go')
    # #max_coeff_abs = np.max(np.abs(data_in))
    # m = show_hist(data=data_in)
    # print(regular_hist(data=data_in))
    #
    # print("difference :", np.mean(m[1] - regular_hist(data=data_in)[1]))
    print("Coeff. quant difference sum: ", np.sum((data_in - data_out)))
    print("Coeff. quant difference mean: ", np.mean((data_in - data_out)))

    return data_out


def re_fexp_cuda(data_in, word_len, frac_len):
    """
    Pytorch implementation of the fixed-point casting.
    Works with cuda tensors.
    :param data_in:
    :param frac_len: bits
    :param enable:
    :return:
    """
    #frac_len = torch.Tensor(np.array([frac_len])).cuda()
    #word_len = torch.from_numpy(np.array([word_len])).cuda()

    if frac_len < 0:
        scale_factor = 1/cuda_clip_table[-frac_len]
    else:
        scale_factor = cuda_clip_table[frac_len]

    scale_factor_inv = 1/scale_factor

    data_out = data_in * scale_factor
    data_out = torch.round(data_out)

    # Sauration limit
    sat_limit = clip_table[word_len]  #  i.e. clip_table((word_len-1) + 1);

    # cuda
    cuda_clipped = torch.clamp(data_out, (1 - sat_limit), (sat_limit - 1))
    cuda_clipped *= scale_factor_inv

    return cuda_clipped


def re_fexp_hist(data_in, word_len, frac_len):
    """
    Pure Python implementation of the fixed-point casting with histogram.
    :type frac_len: int
    :param data_in:
    :param frac_len: bits
    :param enable:
    :return:
    """

    h = np.histogram(data_in.flatten(), normed=False, bins=pow(2, word_len))
    ratio = 1 - (h[0][0] + h[0][-1])/float(data_in.size)  # compare border values

    #  compare values in the border bins (if they are small - reduce)
    if ratio > 0.99:
        frac_len += 1

    if frac_len < 0:
        scale_factor = 1/clip_table[-frac_len]
    else:
        scale_factor = clip_table[frac_len]

    scale_factor_inv = 1/scale_factor

    data_out = np.round(data_in*scale_factor)

    # Sauration limit
    sat_limit = clip_table[word_len]  #  i.e. clip_table((word_len-1) + 1);

    np.clip(data_out, (1 - sat_limit), (sat_limit - 1), data_out)
    # print "python clipped: ", data_out

    data_out *= scale_factor_inv

    return data_out


def re_fexp_hist_cuda(data_in, word_len, frac_len):
    """
    Pytorch implementation of the fixed-point casting.
    Works with cuda tensors.
    :param data_in:
    :param frac_len: bits
    :param enable:
    :return:
    """
    #np_data = data_in.data.cpu().numpy()
    #h = np.histogram(np_data.flatten(), normed=False, bins=pow(2, word_len))
    #ratio_1 = 1 - (h[0][0] + h[0][-1])/float(np_data.size)  # compare border values
    
    #h = torch.histc(data_in.cpu(), bins=pow(2, word_len))
    #ratio_2 = 1 - (h[0] + h[-1])/float(data_in.numel())
    
    bins = 1 << word_len  # pow(2, word_len)
    data_min = data_in.min()
    data_max = data_in.max()
    bin_width = (data_max - data_min) / float(bins)
    count_min = torch.sum(data_in.lt(data_min + bin_width)).float()
    count_max = torch.sum(data_in.ge(data_max - bin_width)).float()
    ratio = 1 - (count_min + count_max) / float(data_in.numel())
    
    #  compare values in the border bins (if they are small - reduce)
    if ratio > 0.99:
        frac_len += 1

    if frac_len < 0:
        scale_factor = 1/cuda_clip_table[-frac_len]
    else:
        scale_factor = cuda_clip_table[frac_len]

    scale_factor = torch.tensor(scale_factor, device=torch.device('cuda', data_in.get_device()))
    scale_factor_inv = 1/scale_factor

    data_out = data_in * scale_factor
    data_out = torch.round(data_out)

    # Sauration limit
    sat_limit = cuda_clip_table[word_len]  #  i.e. clip_table((word_len-1) + 1);

    # cuda
    cuda_clipped = torch.clamp(data_out, (1 - sat_limit), (sat_limit - 1))
    cuda_clipped *= scale_factor_inv

    return cuda_clipped
