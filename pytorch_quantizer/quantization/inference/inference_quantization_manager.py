import torch
import torch.nn as nn
from pytorch_quantizer.quantization import qtypes
from utils.misc import Singleton
from utils import attacher
from utils.monitor import Monitor
from .statistic_manager import StatisticManager
from pytorch_quantizer.quantization.quantization_manager import QuantizationManagerBase
from enum import Enum
from itertools import count
import os
import numpy as np
from utils.dump_manager import DumpManager as DM


class StatsMode(Enum):
    no_stats = 1
    collect_stats = 2
    use_stats = 3


class Conv2dWithId(nn.Conv2d):
    _id = count(0)
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWithId, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.id = next(self._id)
        # print('conv_%d' % self.id)

    # TODO: handle quantization of activations per model
    def forward(self, input):
        activation_id = 'conv%d_activation' % self.id
        if not QMI().enabled:
            out = super(Conv2dWithId, self).forward(input)
        else:
            if QMI().stats_mode is not StatsMode.collect_stats:
                self.weight.data = QMI().quantize_instant(self.weight, "weight")
                if self.bias is not None:
                    self.bias.data = QMI().quantize_instant(self.bias, "bias")
            out = super(Conv2dWithId, self).forward(input)
            if QMI().stats_mode is StatsMode.collect_stats:
                QMI().stats_manager.save_tensor_stats(out, activation_id)
            elif QMI().stats_mode is StatsMode.use_stats:
                # Quantize using statistics
                out = QMI().quantize_instant(out, "activation", stat_id=activation_id)
            else:
                # No stats, quantize using actual values
                out = QMI().quantize_instant(out, "activation")

        return out


class LinearWithId(nn.Linear):
    _id = count(0)
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWithId, self).__init__(in_features, out_features, bias)
        self.id = next(self._id)

    def forward(self, input):
        if not QMI().enabled:
            return super(LinearWithId, self).forward(input)
        else:
            if QMI().stats_mode is not StatsMode.collect_stats:
                self.weight.data = QMI().quantize_instant(self.weight, "weight")
                if self.bias is not None:
                    self.bias.data = QMI().quantize_instant(self.bias, "bias")
            out = super(LinearWithId, self).forward(input)
            activation_id = 'linear%d_activation' % self.id
            if QMI().stats_mode is StatsMode.collect_stats:
                QMI().stats_manager.save_tensor_stats(out, activation_id)
            elif QMI().stats_mode is StatsMode.use_stats:
                out = QMI().quantize_instant(out, "activation_linear", stat_id=activation_id)
            else:
                out = QMI().quantize_instant(out, "activation_linear")
            return out


# TODO: batch norm folding
class BatchNorm2dWithId(nn.BatchNorm2d):
    _id = count(0)
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2dWithId, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.id = next(self._id)
        # print('bn_%d' % self.id)

    def forward(self, input):
        if not QMI().enabled:
            return super(BatchNorm2dWithId, self).forward(input)
        else:
            if QMI().bn_folding:
                # Do regular BN if floding is set
                return super(BatchNorm2dWithId, self).forward(input)

            if QMI().stats_mode is not StatsMode.collect_stats:
                self.weight.data = QMI().quantize_instant(self.weight, "weight")
                if self.bias is not None:
                    self.bias.data = QMI().quantize_instant(self.bias, "bias")

            out = super(BatchNorm2dWithId, self).forward(input)
            activation_id = 'bn%d_activation' % self.id
            if QMI().stats_mode is StatsMode.collect_stats:
                QMI().stats_manager.save_tensor_stats(out, activation_id)
            elif QMI().stats_mode is StatsMode.use_stats:
                # Quantize using statistics
                out = QMI().quantize_instant(out, "activation", stat_id=activation_id)
            else:
                # No stats, quantize using actual values
                out = QMI().quantize_instant(out, "activation")
            return out


class QuantizationManagerInference(QuantizationManagerBase):
    def __init__(self, args, qparams):
        super(QuantizationManagerInference, self).__init__()
        self.quantize = args.qtype is not None
        if self.quantize:
            print("Quantize to %s" % args.qtype)
        self.op_manager = self.createTruncationManager(args, qparams)
        self.enabled = False
        self.bn_folding = False
        sf = args.stats_folder if args.stats_folder is not None else args.arch
        if args.stats_mode == 'collect':
            self.stats_mode = StatsMode.collect_stats
            self.stats_manager = StatisticManager(sf, load_stats=False)
        elif args.stats_mode == 'use':
            self.stats_mode = StatsMode.use_stats
            self.stats_manager = StatisticManager(sf, load_stats=True)
        else:
            self.stats_mode = StatsMode.no_stats
            self.stats_manager = None

    def __exit__(self, *args):
        if self.stats_manager is not None:
            self.stats_manager.__exit__()
        super(QuantizationManagerInference, self).__exit__(args)

    def createTruncationManager(self, args, qparams):
        op_manager = TruncationOpManagerInference(args, qparams)
        op_manager.set_8bit_list(['conv0_activation'])
        return op_manager

    def quantize_instant(self, tensor, tag="", stat_id=None):
        return self.op_manager.quantize_instant(tensor, tag, stat_id)

    def set_8bit_list(self, ignore_ids):
        self.op_manager.set_8bit_list(ignore_ids)


# Alias
QMI = QuantizationManagerInference


class TruncationOpManagerInference:
    def __load_quantizer__(self, qtype, qparams):
        qtype_name = qtype.rstrip('1234567890')
        quant_params = qparams[qtype_name] if qtype_name in qparams else {}
        quantizer = qtypes.__dict__[qtype_name + "_quantizer"](qtype, quant_params)
        return quantizer, quant_params

    def __init__(self, args, qparams):
        self.verbose = False
        self.activation_quantizer = None

        self.origin_linear = nn.Linear
        self.origin_conv2d = nn.Conv2d
        self.origin_batch_norm = nn.BatchNorm2d

        if args.qtype is not None:
            self.quantize = True
            self.activation_quantizer, _ = self.__load_quantizer__(args.qtype, qparams)
            self.linear_layer_quantizer, _ = self.__load_quantizer__('int8', qparams)
            # self.weights_quantizer = self.activation_quantizer
            self.weights_quantizer, _ = self.__load_quantizer__('int8', qparams)
            self.quantizer_4bit, _ = self.__load_quantizer__('int4', qparams)
            self.quantizer_8bit, _ = self.__load_quantizer__('int8', qparams)

    def set_8bit_list(self, ignore_list):
        self.ignore_ids = ignore_list

    def enable(self):
        # self.quantize_matmul()
        self.quantize_linear()
        self.quantize_conv2d()
        self.quantize_batch_norm()

    def disable(self):
        nn.Linear = self.origin_linear
        nn.Conv2d = self.origin_conv2d
        nn.BatchNorm2d = self.origin_batch_norm

    # quantizes origin matmul
    def quantize_matmul(self):
        def quantized_matmul(tensor1, tensor2):
            tensor1_ = attacher.pytorch_attach(tensor1, self.activation_quantizer, None)
            tensor2_ = attacher.pytorch_attach(tensor2, self.activation_quantizer, None)
            res = self.origin_matmul(tensor1_, tensor2_)
            return attacher.pytorch_attach(res, self.activation_quantizer, None)

        torch.Tensor.matmul = quantized_matmul

    # quantizes origin linear
    def quantize_linear(self):
        nn.Linear = LinearWithId

    # quantizes origin conv2d
    def quantize_conv2d(self):
        nn.Conv2d = Conv2dWithId

    def quantize_batch_norm(self):
        nn.BatchNorm2d = BatchNorm2dWithId


    def quantize_tensor(self, tensor, fprop=True, bprop=True):
        fprop = self.activation_quantizer if fprop else None
        return attacher.pytorch_attach(tensor, fprop, None)

    def quantize_instant(self, tensor, tag="", stat_id=None):
        # ignore quantization of first and last layer
        ignore_cond = False
        if stat_id is not None:
            ignore_cond = np.array([l == stat_id for l in self.ignore_ids]).any()
        if ignore_cond:
            return self.quantizer_8bit(tensor, tag, stat_id)
        # Leave classifier layer in 8 bit
        elif tag == 'activation_linear' and tensor.shape[1] == 1000:
            return self.linear_layer_quantizer(tensor, tag, stat_id)
        elif tag == 'activation':
            return self.activation_quantizer(tensor, tag, stat_id)
        else: # weight, bias
            return self.weights_quantizer(tensor, tag, stat_id)
