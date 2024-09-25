import math
import torch
from torch import nn


from ..basic_routines.quantization_routines import linear_, linear_hist, asymmetric_
from ..basic_routines.quantization_routines import compute_integral_part
from ..basic_routines.quantization_routines import mean_for_asymmetric, scale_for_asymmetric, integral_part_for_asymmetric
import pdb

class QuantLayer(nn.Module):
    def __init__(self,
                 word_length=8,
                 target_layer_name=None,
                 in_channels=None,
                 out_channels=None,
                 weight_shape=None,
                 quantization_procedure=linear_):
        super(QuantLayer, self).__init__()
        self.word_length = word_length
        self.target_layer_name = target_layer_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_shape = weight_shape
        self.quantization_procedure = quantization_procedure
        self.register_buffer('frac', torch.tensor(1024))
        self.stats_only = True
        self.transparent = True

    def forward(self, data_in):
        if self.transparent:
            return data_in

        if self.stats_only:
            if self.quantization_procedure in [linear_, linear_hist]:
                frac = torch.tensor(self.word_length - 1 - compute_integral_part(data_in=data_in), device=torch.device('cuda', data_in.get_device()))
                #pdb.set_trace()
                #if data_in.is_cuda:
                #    self.frac = self.frac.cuda()
                self.frac.fill_(torch.min(self.frac, frac))
            return data_in
        else:

            # quantization routine
            if self.quantization_procedure in [linear_, linear_hist]:
                output = self.quantization_procedure(data_in=data_in,
                                                     frac_len=self.frac.item(),
                                                     word_len=self.word_length)
            return output
