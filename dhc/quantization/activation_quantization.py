from torch import nn
from collections import OrderedDict

from .basic_routines.cadence import linear_cadence
from .basic_routines.gen_quant_layer import QuantLayer

import sys

class ActivationQuantization(object):

    def __init__(self, quantization_procedure=linear_cadence, stats_gathering_limit=4, word_len=8):

        self._stats_gathering_limit = stats_gathering_limit
        self._word_length = word_len
        self._quantization_procedure = quantization_procedure
        self._quantization_layer = QuantLayer

    def get_stats_for_quantization(self, model, name='.'):
        for k, v in model._modules.items():
            if isinstance(v, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                l = OrderedDict()
                l[k] = v
                if (hasattr(v, 'weight')):
                    l['quant_{}'.format(k)] = self._quantization_layer(target_layer_name=name + '.' + k, weight_shape=v.weight.data.numpy().shape, quantization_procedure=self._quantization_procedure, word_length=self._word_length)
                else:
                    l['quant_{}'.format(k)] = self._quantization_layer(target_layer_name=name + '.' + k, weight_shape=None, quantization_procedure=self._quantization_procedure, word_length=self._word_length)
                m = nn.Sequential(l)
                model._modules[k] = m
            else:
                self.get_stats_for_quantization(v, name + '.' + k)
        return model

    def get_quantization_layer(self):
        return self._quantization_layer

    def get_stats_(self, model):
        """

        :param model:
        :return:
        """

        #import pdb
        #pdb.set_trace()

        for k, v in model._modules.items():
            if isinstance(v, QuantLayer):
                v.stats_only = True
                v.frac.fill_(40)
                v.transparent = False
                model._modules[k] = v
            else:
                self.get_stats_(v)
        return model

    def get_quantized_model_(self, model):
        """

        :param model:
        :return:
        """
        for k, v in model._modules.items():
            if isinstance(v, QuantLayer):
                v.stats_only = False
                v.transparent = False
                model._modules[k] = v
            else:
                self.get_quantized_model_(v)
        return model

    def get_quantized_model(self, model):
        """

        :param model:
        :return:
        """
        for k, v in model._modules.items():
            if 'quant' in k:
                v.stats_only = False
                model._modules[k] = v
            else:
                self.get_quantized_model(v)
        return model


    def get_transparent_model(self, model):
        """

        :param model:
        :return:
        """
        for k, v in model._modules.items():
            if 'quant' in k:
                v.transparent = True
                v.stats_only = False
                model._modules[k] = v
            else:
                self.get_transparent_model(v)
        return model
