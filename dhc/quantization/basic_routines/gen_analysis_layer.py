import math
import torch
from torch import nn


from ..basic_routines.cadence import linear_cadence, linear_cadence_hist, asymmetric_cadence
from ..basic_routines.cadence import compute_integral_part
from ..basic_routines.cadence import mean_for_asymmetric, scale_for_asymmetric, integral_part_for_asymmetric
from ..basic_routines.cadence import td_analysis

class AnalysisLayer(nn.Module):
    def __init__(self,
                 target_layer_name=None,
                 in_c=None,
                 out_c=None,
                 stride=None,
                 padding=None,
                 kernel=None,
                 bias=None,
                 weight_shape=None,
                 analysis_procedure=td_analysis,
                 bins=100):
        super(AnalysisLayer, self).__init__()
        self.target_layer_name = target_layer_name
        self.in_channels = in_c
        self.out_channels = out_c
        self.padding = padding
        self.stride = stride
        self.kernel = kernel
        self.bias = bias
        self.weight_shape = weight_shape
        self.analysis_procedure = analysis_procedure
        #self.register_buffer('frac', torch.tensor(1024))
        self.register_buffer('td_hist', torch.zeros((100, 100), dtype=torch.float32))
        self.register_buffer('td_energy', torch.zeros((out_c, in_c)))
        self.register_buffer('td_energy_3d', torch.zeros((out_c, )))
        self.stats_only = True

    def forward(self, data_in):
        if self.stats_only:
            if self.analysis_procedure in [td_analysis]:
                
                #for i in range():
                #    for j in range():
                #        for k in range():
                #            for m in range():
                #                for x in range():
                #                    for y in range():
                     
                #print(data_in.size())   
                #print(self.td_energy_3d.size())
                #print(self.out_channels)

                for i in range(self.out_channels):
                    self.td_energy_3d[i] = self.td_energy_3d[i] + torch.mean(torch.abs(data_in[:,i,:,:].flatten())).item()
                    #print(torch.mean(torch.abs(data_in[:,i,:,:].flatten())))

                self.td_hist = torch.add(self.td_hist, torch.ones((100, 100)).cuda())
                #self.td_hist.fill_(torch.add(self.td_hist, torch.ones((100, 100)).cuda()))
                #frac = torch.tensor(self.word_length - 1 - compute_integral_part(data_in=data_in), device=torch.device('cuda', data_in.get_device()))
                #self.frac.fill_(torch.min(self.frac, frac))
            return data_in
        else:
            # quantization routine
            if self.quantization_procedure in [linear_cadence, linear_cadence_hist]:
                output = self.quantization_procedure(data_in=data_in,
                                                     frac_len=self.frac.item(),
                                                     word_len=self.word_length)
            return output
