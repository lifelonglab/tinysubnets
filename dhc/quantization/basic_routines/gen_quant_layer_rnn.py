import math
import torch
from torch import nn
import numpy as np
import pdb

from ..basic_routines.cadence import linear_cadence, linear_cadence_hist, asymmetric_cadence
from ..basic_routines.cadence import compute_integral_part
from ..basic_routines.cadence import mean_for_asymmetric, scale_for_asymmetric, integral_part_for_asymmetric

class QuantLayer(nn.Module):
    def __init__(self,
                 name, quantization_procedure=linear_cadence_hist,
                 word_length=8, percentile=0.999, scaler_bits=8, qlstm=False):
        super(QuantLayer, self).__init__()
        self.word_length = word_length
        self.avg = 0
        self.overflow_rate = 0.0
        self.name = name
        self.quantization_procedure = quantization_procedure
        self.register_buffer('frac', torch.tensor(1024))
        self.register_buffer('percentile', torch.tensor(1024))
        self.register_buffer('hist_min', torch.tensor(2048.0))
        self.register_buffer('hist_max', torch.tensor(-2048.0))
        self.register_buffer('histVal', torch.zeros(1000))
        self.register_buffer('histNum', torch.tensor(0.0))
        self.register_buffer('cumsumhist', torch.zeros(1000))
        self.register_buffer('histbins', torch.zeros(1001))
        self.register_buffer('clipThresh', torch.tensor(0.999))
        self.register_buffer('scale', torch.tensor(1024))
        self.register_buffer('scaleQuant', torch.tensor(1024.0))
        self.register_buffer('scaleDiff', torch.tensor(1024.0))
        self.register_buffer('scaler_bits', torch.tensor(1024))
        self.register_buffer('sf_scaler', torch.tensor(1024))
        
        self.qlstm = qlstm
        self.stats_only = True
        self.sim_mode = 0
        self.np2 = False
        self.numBins = 1000
        self.use_scale = 0
        self.batchSize = 32
        #self.scaleQuant = 8
        #self.scaleDiff = 8 
        #self.sf_scaler = 8
        #self.scaler_bits = 8 

    def forward(self, data_in):
        
        #if self.stats_only:
        if self.sim_mode == 0: 
            if self.quantization_procedure in [linear_cadence, linear_cadence_hist]:
                #frac = torch.tensor(self.word_length - 1 - compute_integral_part(data_in=data_in),
                #                    device=torch.device('cuda:1'))
                              
                #self.frac.fill_(torch.min(self.frac, frac))
                #pdb.set_trace()               
                self.hist_min = torch.min(torch.min(data_in.flatten()), self.hist_min)
                self.hist_max = torch.max(torch.max(data_in.flatten()), self.hist_max)
                
            return data_in
        elif self.sim_mode == 1:
            # quantization routine
            if self.quantization_procedure in [linear_cadence, linear_cadence_hist]:
                #output = self.quantization_procedure(data_in=data_in,
                #                                     frac_len=self.frac.item(),
                #                                     word_len=self.word_length)
                #pdb.set_trace()
                histVal_ = np.histogram(data_in.cpu().flatten().numpy(), bins=self.numBins, range=(self.hist_min.cpu().item(), self.hist_max.cpu().item()), normed=True)
                histVal_, bins_ = torch.from_numpy(histVal_[0]), torch.from_numpy(histVal_[1])
                #pdb.set_trace()
                self.histVal = self.histVal + histVal_.cuda(1)
                self.histbins = bins_
                self.histNum = self.histNum + 1*self.batchSize
                #self.histFinal = self.histVal/self.histNum

            return data_in

        elif self.sim_mode == 2:
            #pdb.set_trace()
            self.cumsumhist = np.cumsum(self.histVal.cpu().flatten())/np.sum(self.histVal.cpu().numpy())
            maxidx = np.argmax(self.cumsumhist.flatten() > self.clipThresh.cpu().item()) #(self.cumsumhist > 0.9999)
            lowerIdx = np.argmax(self.cumsumhist.flatten() >= (1-self.clipThresh.cpu().item())) #(self.cumsumhist >= 0.0001)
            self.percentile = torch.tensor(np.float32(np.maximum(np.abs(self.histbins.cpu()[np.int32(maxidx)].item()), np.abs(self.histbins.cpu()[np.int32(lowerIdx)].item()))))
            #pdb.set_trace()
            max_coeff_abs = self.percentile
            if (max_coeff_abs > 0.0):
                int_bits = int(math.ceil(math.log(max_coeff_abs, 2)))
            else:
                int_bits = 0

            self.frac = torch.tensor(self.word_length - 1 - int_bits) #compute_integral_part(self.percentile, self.overflow_rate, mean_val=self.avg)

            if self.use_scale:
                self.scale = ((self.percentile - 0) / math.pow(2, self.word_length - 1 - self.frac))
                if self.scaler_bits < 32:
                    self.sf_scaler = self.scaler_bits - compute_integral_part_arr(self.scale, 0)
                    self.scaleQuant = linear_cadence(self.scale, self.sf_scaler, self.scaler_bits)
                    self.scaleDiff = self.scale - self.scaleQuant

            return data_in

        else:
            # quantization routine
            if 'ForgetByHidden' not in self.name and  self.quantization_procedure in [linear_cadence, linear_cadence_hist]:
                #if (self.sim_mode == 4):
                #    pdb.set_trace()
                output = self.quantization_procedure(data_in=data_in,
                                                     frac_len=self.frac.item(),
                                                     word_len=self.word_length)
            elif 'ForgetByHidden' in self.name:
                output = self.quantization_procedure(data_in=data_in,
                                                     frac_len=15,
                                                     word_len=self.word_length)
            return output


class QuantLayer_common(nn.Module):
    def __init__(self,
                 name, quantization_procedure=linear_cadence_hist,
                 word_length=8, percentile=0.999, scaler_bits=8, qlstm=False):
        super(QuantLayer_common, self).__init__()
        self.word_length = word_length
        self.avg = 0
        self.name = name
        self.overflow_rate = 0.0
        self.quantization_procedure = quantization_procedure

        self.register_buffer('frac1', torch.tensor(1024))
        self.register_buffer('percentile1', torch.tensor(1024))
        self.register_buffer('hist_min1', torch.tensor(2048.0))
        self.register_buffer('hist_max1', torch.tensor(-2048.0))
        self.register_buffer('histVal1', torch.zeros(1000))
        self.register_buffer('histNum1', torch.tensor(0.0))
        self.register_buffer('cumsumhist1', torch.zeros(1000))
        self.register_buffer('histbins1', torch.zeros(1001))
        self.register_buffer('clipThresh', torch.tensor(0.999))
        self.register_buffer('scale1', torch.tensor(1024))
        self.register_buffer('scaleQuant1', torch.tensor(1024.0))
        self.register_buffer('scaleDiff1', torch.tensor(1024.0))
        self.register_buffer('scaler_bits1', torch.tensor(1024))
        self.register_buffer('sf_scaler1', torch.tensor(1024))

        self.register_buffer('frac2', torch.tensor(1024))
        self.register_buffer('percentile2', torch.tensor(1024))
        self.register_buffer('hist_min2', torch.tensor(2048.0))
        self.register_buffer('hist_max2', torch.tensor(-2048.0))
        self.register_buffer('histVal2', torch.zeros(1000))
        self.register_buffer('histNum2', torch.tensor(0.0))
        self.register_buffer('cumsumhist2', torch.zeros(1000))
        self.register_buffer('histbins2', torch.zeros(1001))
        self.register_buffer('clipThresh2', torch.tensor(0.999))
        self.register_buffer('scale2', torch.tensor(1024))
        self.register_buffer('scaleQuant2', torch.tensor(1024.0))
        self.register_buffer('scaleDiff2', torch.tensor(1024.0))
        self.register_buffer('scaler_bits2', torch.tensor(1024))
        self.register_buffer('sf_scaler2', torch.tensor(1024))

        self.register_buffer('frac3', torch.tensor(1024))
        self.register_buffer('percentile3', torch.tensor(1024))
        self.register_buffer('hist_min3', torch.tensor(2048.0))
        self.register_buffer('hist_max3', torch.tensor(-2048.0))
        self.register_buffer('histVal3', torch.zeros(1000))
        self.register_buffer('histNum3', torch.tensor(0.0))
        self.register_buffer('cumsumhist3', torch.zeros(1000))
        self.register_buffer('histbins3', torch.zeros(1001))
        self.register_buffer('clipThresh3', torch.tensor(0.999))
        self.register_buffer('scale3', torch.tensor(1024))
        self.register_buffer('scaleQuant3', torch.tensor(1024.0))
        self.register_buffer('scaleDiff3', torch.tensor(1024.0))
        self.register_buffer('scaler_bits3', torch.tensor(1024))
        self.register_buffer('sf_scaler3', torch.tensor(1024))

        self.register_buffer('frac4', torch.tensor(1024))
        self.register_buffer('percentile4', torch.tensor(1024))
        self.register_buffer('hist_min4', torch.tensor(2048.0))
        self.register_buffer('hist_max4', torch.tensor(-2048.0))
        self.register_buffer('histVal4', torch.zeros(1000))
        self.register_buffer('histNum4', torch.tensor(0.0))
        self.register_buffer('cumsumhist4', torch.zeros(1000))
        self.register_buffer('histbins4', torch.zeros(1001))
        self.register_buffer('clipThresh4', torch.tensor(0.999))
        self.register_buffer('scale4', torch.tensor(1024))
        self.register_buffer('scaleQuant4', torch.tensor(1024.0))
        self.register_buffer('scaleDiff4', torch.tensor(1024.0))
        self.register_buffer('scaler_bits4', torch.tensor(1024))
        self.register_buffer('sf_scaler4', torch.tensor(1024))
        
        self.qlstm = qlstm
        self.stats_only = True
        self.sim_mode = 0
        self.np2 = False
        self.numBins = 1000
        self.use_scale = 0
        self.batchSize = 32
        #self.scaleQuant = 8
        #self.scaleDiff = 8 
        #self.sf_scaler = 8
        #self.scaler_bits = 8 

    def forward(self, data_in):
        
        chunk = int(data_in.size()[1]/4)
        #if self.stats_only:
        if self.sim_mode == 0: 
            if self.quantization_procedure in [linear_cadence, linear_cadence_hist]:
                #frac = torch.tensor(self.word_length - 1 - compute_integral_part(data_in=data_in),
                #                    device=torch.device('cuda:1'))
                              
                #self.frac.fill_(torch.min(self.frac, frac))
                #pdb.set_trace()
                self.hist_min1 = torch.min(torch.min(data_in[:,:chunk].flatten()), self.hist_min1)
                self.hist_max1 = torch.max(torch.max(data_in[:,:chunk].flatten()), self.hist_max1)
                self.hist_min2 = torch.min(torch.min(data_in[:,chunk:2*chunk].flatten()), self.hist_min2)
                self.hist_max2 = torch.max(torch.max(data_in[:,chunk:2*chunk].flatten()), self.hist_max2)
                self.hist_min3 = torch.min(torch.min(data_in[:,2*chunk:3*chunk].flatten()), self.hist_min3)
                self.hist_max3 = torch.max(torch.max(data_in[:,2*chunk:3*chunk].flatten()), self.hist_max3)
                self.hist_min4 = torch.min(torch.min(data_in[:,3*chunk:].flatten()), self.hist_min4)
                self.hist_max4 = torch.max(torch.max(data_in[:,3*chunk:].flatten()), self.hist_max4)
                
            return data_in
        elif self.sim_mode == 1:
            # quantization routine
            if self.quantization_procedure in [linear_cadence, linear_cadence_hist]:
                #output = self.quantization_procedure(data_in=data_in,
                #                                     frac_len=self.frac.item(),
                #                                     word_len=self.word_length)
                #pdb.set_trace()
                histVal_ = np.histogram(data_in[:,:chunk].cpu().flatten().numpy(), bins=self.numBins, range=(self.hist_min1.cpu().item(), self.hist_max1.cpu().item()), normed=True)
                histVal_, bins_ = torch.from_numpy(histVal_[0]), torch.from_numpy(histVal_[1])
                #pdb.set_trace()
                self.histVal1 = self.histVal1 + histVal_.cuda(1)
                self.histbins1 = bins_
                self.histNum1 = self.histNum1 + 1*self.batchSize
                #self.histFinal = self.histVal/self.histNum

                histVal_ = np.histogram(data_in[:,chunk:2*chunk].cpu().flatten().numpy(), bins=self.numBins, range=(self.hist_min2.cpu().item(), self.hist_max2.cpu().item()), normed=True)
                histVal_, bins_ = torch.from_numpy(histVal_[0]), torch.from_numpy(histVal_[1])
                #pdb.set_trace()
                self.histVal2 = self.histVal2 + histVal_.cuda(1)
                self.histbins2 = bins_
                self.histNum2 = self.histNum2 + 1*self.batchSize
                #self.histFinal = self.histVal/self.histNum

                histVal_ = np.histogram(data_in[:,2*chunk:3*chunk].cpu().flatten().numpy(), bins=self.numBins, range=(self.hist_min3.cpu().item(), self.hist_max3.cpu().item()), normed=True)
                histVal_, bins_ = torch.from_numpy(histVal_[0]), torch.from_numpy(histVal_[1])
                #pdb.set_trace()
                self.histVal3 = self.histVal3 + histVal_.cuda(1)
                self.histbins3 = bins_
                self.histNum3 = self.histNum3 + 1*self.batchSize
                #self.histFinal = self.histVal/self.histNum
 
                histVal_ = np.histogram(data_in[:,3*chunk:].cpu().flatten().numpy(), bins=self.numBins, range=(self.hist_min4.cpu().item(), self.hist_max4.cpu().item()), normed=True)
                histVal_, bins_ = torch.from_numpy(histVal_[0]), torch.from_numpy(histVal_[1])
                #pdb.set_trace()
                self.histVal4 = self.histVal4 + histVal_.cuda(1)
                self.histbins4 = bins_
                self.histNum4 = self.histNum4 + 1*self.batchSize
                #self.histFinal = self.histVal/self.histNum

            return data_in

        elif self.sim_mode == 2:
            #pdb.set_trace()
            self.cumsumhist1 = np.cumsum(self.histVal1.cpu().flatten())/np.sum(self.histVal1.cpu().numpy())
            maxidx = np.argmax(self.cumsumhist1.flatten() > self.clipThresh.cpu().item()) #(self.cumsumhist > 0.9999)
            lowerIdx = np.argmax(self.cumsumhist1.flatten() >= (1-self.clipThresh.cpu().item())) #(self.cumsumhist >= 0.0001)
            self.percentile = torch.tensor(np.float32(np.maximum(np.abs(self.histbins1.cpu()[np.int32(maxidx)].item()), np.abs(self.histbins1.cpu()[np.int32(lowerIdx)].item()))))
            #pdb.set_trace()
            max_coeff_abs = self.percentile
            if (max_coeff_abs > 0.0):
                int_bits = int(math.ceil(math.log(max_coeff_abs, 2)))
            else:
                int_bits = 0

            self.frac1 = torch.tensor(self.word_length - 1 - int_bits) #compute_integral_part(self.percentile, self.overflow_rate, mean_val=self.avg)

            if self.use_scale:
                self.scale = ((self.percentile - 0) / math.pow(2, self.word_length - 1 - self.frac))
                if self.scaler_bits < 32:
                    self.sf_scaler = self.scaler_bits - compute_integral_part_arr(self.scale, 0)
                    self.scaleQuant = linear_cadence(self.scale, self.sf_scaler, self.scaler_bits)
                    self.scaleDiff = self.scale - self.scaleQuant

            #pdb.set_trace()
            self.cumsumhist2 = np.cumsum(self.histVal2.cpu().flatten())/np.sum(self.histVal2.cpu().numpy())
            maxidx = np.argmax(self.cumsumhist2.flatten() > self.clipThresh.cpu().item()) #(self.cumsumhist > 0.9999)
            lowerIdx = np.argmax(self.cumsumhist2.flatten() >= (1-self.clipThresh.cpu().item())) #(self.cumsumhist >= 0.0001)
            self.percentile = torch.tensor(np.float32(np.maximum(np.abs(self.histbins2.cpu()[np.int32(maxidx)].item()), np.abs(self.histbins2.cpu()[np.int32(lowerIdx)].item()))))
            #pdb.set_trace()
            max_coeff_abs = self.percentile
            if (max_coeff_abs > 0.0):
                int_bits = int(math.ceil(math.log(max_coeff_abs, 2)))
            else:
                int_bits = 0

            self.frac2 = torch.tensor(self.word_length - 1 - int_bits) #compute_integral_part(self.percentile, self.overflow_rate, mean_val=self.avg)

            #pdb.set_trace()
            self.cumsumhist3 = np.cumsum(self.histVal3.cpu().flatten())/np.sum(self.histVal3.cpu().numpy())
            maxidx = np.argmax(self.cumsumhist3.flatten() > self.clipThresh.cpu().item()) #(self.cumsumhist > 0.9999)
            lowerIdx = np.argmax(self.cumsumhist3.flatten() >= (1-self.clipThresh.cpu().item())) #(self.cumsumhist >= 0.0001)
            self.percentile = torch.tensor(np.float32(np.maximum(np.abs(self.histbins3.cpu()[np.int32(maxidx)].item()), np.abs(self.histbins3.cpu()[np.int32(lowerIdx)].item()))))
            #pdb.set_trace()
            max_coeff_abs = self.percentile
            if (max_coeff_abs > 0.0):
                int_bits = int(math.ceil(math.log(max_coeff_abs, 2)))
            else:
                int_bits = 0

            self.frac3 = torch.tensor(self.word_length - 1 - int_bits) #compute_integral_part(self.percentile, self.overflow_rate, mean_val=self.avg)

            #pdb.set_trace()
            self.cumsumhist4 = np.cumsum(self.histVal4.cpu().flatten())/np.sum(self.histVal4.cpu().numpy())
            maxidx = np.argmax(self.cumsumhist4.flatten() > self.clipThresh.cpu().item()) #(self.cumsumhist > 0.9999)
            lowerIdx = np.argmax(self.cumsumhist4.flatten() >= (1-self.clipThresh.cpu().item())) #(self.cumsumhist >= 0.0001)
            self.percentile = torch.tensor(np.float32(np.maximum(np.abs(self.histbins4.cpu()[np.int32(maxidx)].item()), np.abs(self.histbins4.cpu()[np.int32(lowerIdx)].item()))))
            #pdb.set_trace()
            max_coeff_abs = self.percentile
            if (max_coeff_abs > 0.0):
                int_bits = int(math.ceil(math.log(max_coeff_abs, 2)))
            else:
                int_bits = 0

            self.frac4 = torch.tensor(self.word_length - 1 - int_bits) #compute_integral_part(self.percentile, self.overflow_rate, mean_val=self.avg)

            return data_in

        else:
            # quantization routine
            #if (self.sim_mode == 4):
            #    pdb.set_trace()
            if self.quantization_procedure in [linear_cadence, linear_cadence_hist]:
                #pdb.set_trace()
                output1 = self.quantization_procedure(data_in=data_in[:,:chunk],
                                                     frac_len=self.frac1.item(),
                                                     word_len=self.word_length)

                output2 = self.quantization_procedure(data_in=data_in[:,chunk:2*chunk],
                                                     frac_len=self.frac2.item(),
                                                     word_len=self.word_length)

                output3 = self.quantization_procedure(data_in=data_in[:,2*chunk:3*chunk],
                                                     frac_len=self.frac3.item(),
                                                     word_len=self.word_length)

                output4 = self.quantization_procedure(data_in=data_in[:,3*chunk:],
                                                     frac_len=self.frac4.item(),
                                                     word_len=self.word_length)

            return torch.cat([output1, output2, output3, output4], dim=1)


class QuantLOG_SA(nn.Module):
    def __init__(self, name, after, quantization_procedure=linear_cadence_hist, word_length=8, frac=15):
        super(QuantLOG_SA, self).__init__()
        self.frac = frac
        self.after = after
        self.word_length = word_length 
        self.quantization_procedure = quantization_procedure
        self.sim_mode = 0
        self.name = name

    def forward(self, data_in):
        if (self.sim_mode == 4):
            #pdb.set_trace()
            output = self.quantization_procedure(data_in=data_in, frac_len=self.frac, word_len=self.word_length)
            return output
        else:
            return data_in
    
