import torch
from collections import OrderedDict

from .basic_routines.cadence import linear_cadence, linear_cadence_hist
from .basic_routines.standard_methods import wrapped_hist_quant_adaptive, wrapped_cluster_kmeans
from .basic_routines.cadence import compute_integral_part, asymmetric_cadence
from .basic_routines.cadence import mean_for_asymmetric, scale_for_asymmetric, integral_part_for_asymmetric
import numpy as np
import pdb

class ParametersQuantization(object):

    def __init__(self, quantization_procedure=linear_cadence, word_len=8):
        self._quantization_procedure = quantization_procedure
        self._word_length = word_len

    def get_quantized_model(self, model, task_id=-1):
        """
        Use the dynamic-fixed point for the quantization.
        :param bits: number of target bits
        :return:
        """

        frac = {}
        int_part = {}

        state_dict = model.state_dict()
        #convs = []
        convs = [k for k in state_dict.keys() if 'conv' in k or 'rnn' in k]
        quant_key = ['conv']
        if len(convs) == 0:
            quant_key = ['weight', 'bias']
        state_dict_quant = OrderedDict()
        counter = 0
        #pdb.set_trace()

        for k, v in state_dict.items():
            if 'last' in k and str(task_id) in k:
                frac_len = self._word_length - 1 - compute_integral_part(data_in=v)
                frac[k] = np.power(2.0, -frac_len)
                int_part[k] = np.power(2.0, compute_integral_part(data_in=v))  
                    
                quant_layer = self._quantization_procedure(data_in=v, frac_len=frac_len, word_len=self._word_length)

            if any(q_k in k for q_k in quant_key):
                if self._quantization_procedure == linear_cadence:  # just for linear cadence
                    #import pdb
                    #pdb.set_trace()
                    
                    frac_len = self._word_length - 1 - compute_integral_part(data_in=v)
                    frac[k] = np.power(2.0, -frac_len)
                    int_part[k] = np.power(2.0, compute_integral_part(data_in=v))  
                    
                    #print(frac_len)
                    #print(np.power(2.0, -frac_len))
                    #print(compute_integral_part(data_in=v))
                    quant_layer = self._quantization_procedure(
                        data_in=v,
                        frac_len=frac_len,
                        word_len=self._word_length)
                elif self._quantization_procedure == wrapped_hist_quant_adaptive:
                    quant_layer = self._quantization_procedure(
                        data_in=v,
                        word_len=self._word_length
                    )
                elif self._quantization_procedure == wrapped_cluster_kmeans:
                    quant_layer = self._quantization_procedure(
                        data_in=v,
                        word_len=self._word_length
                    )
                elif self._quantization_procedure == linear_cadence_hist:
                    frac_len = self._word_length - 1 - compute_integral_part(data_in=v)
                    quant_layer = self._quantization_procedure(
                        data_in=v,
                        frac_len=frac_len,
                        word_len=self._word_length)
                elif self._quantization_procedure == asymmetric_cadence:
                    mean_val = mean_for_asymmetric(data_in=v)
                    print("mean values:", mean_val)
                    sf = integral_part_for_asymmetric(
                        input_value=v,
                        mean_val=mean_val
                    )
                    print("int values:", sf)
                    scale_val = scale_for_asymmetric(data_in=v, integer_bits=sf)
                    print("scale values:", scale_val)
                    quant_layer = self._quantization_procedure(
                        input_value=v,
                        sf=sf,
                        bits=self._word_length,
                        mean_val=mean_val,
                        scale_val=scale_val)
                    print("v mean: ", v[0])
                    print("quant_layer: ", quant_layer[0])
                else:
                    quant_layer = self._quantization_procedure(
                        data_in=v.numpy(),
                        word_len=self._word_length) 
            else:
                quant_layer = v

            if any(q_k in k for q_k in quant_key):
                counter += 1

            state_dict_quant[k] = quant_layer

        model.load_state_dict(state_dict_quant)
        print('Quantization: {} bits, procedure: {} '.format(self._word_length, str(self._quantization_procedure.__name__)))
        return model, frac, int_part