import torch
from torch.autograd import Function
import numpy as np
import int_quantization
import math
from utils.monitor import Monitor
from pytorch_quantizer.quantization.inference.statistic_manager import StatisticManager as SM

# Alpha coeficients for for gaussian clipping
# [1.71063519 2.15159277 2.55913646 2.93620062 3.28691474 3.6151146 3.92403714]

# Alpha coeficients for for laplace clipping
# [2.83068299 3.89722946 5.02864014 6.20476633 7.41312622 8.64561995 9.89675982]


count = 0
class IntQuantizer(Function):
    def __init__(self, size, params):
        self.num_bits = size
        # TODO: expose as cmd line parameters
        self.stochastic = False
        self.int_exp = False
        self.enforce_true_zero = True #params['true_zero']
        self.clipping = params['threshold']
        self.alpha_gaus = {2: 1.71, 3: 2.15, 4: 2.55, 5: 2.93, 6: 3.28, 7: 3.61, 8: 3.92}
        self.alpha_laplace = {2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89}
        self.gaussian_const = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) ** 0.5)

    def __call__(self, tensor, tag="", stat_id=None):
        if self.clipping == 'no' or tag != 'activation':
            min_, max_, mean_ = None, None, None
            if stat_id is not None:
                # kind = {'min': 'mean', 'max': 'mean', 'mean': 'mean', 'std': 'mean', 'range': 'mean', 'mean_abs': 'mean', 'b': 'mean'}
                kind = {'min': 'mean', 'max': 'mean', 'mean': 'mean', 'std': 'mean', 'range': 'mean', 'mean_abs': 'mean', 'b': 'mean'}
                # Hack: handle classifier layer differently
                if tag == 'activation_linear' and tensor.shape[1] == 1000:
                    kind['min'] = 'min'
                    kind['max'] = 'max'
                    kind['range'] = 'max'
                min_, max_, mean_, _, _, _, _ = SM().get_tensor_stats(stat_id, kind)
                # print("use stats  for %s, min %f, max %f" % (stat_id, min_, max_))
            return self.gemmlowpQuantize(tensor, min_value=min_, max_value=max_, mean=mean_)
        else:
            return self.gemmlowpClippingQuantize(tensor, tag, stat_id=stat_id, clip_type=self.clipping)

    def get_alpha_laplace(self, tensor, stat_id=None):
        if stat_id is not None:
            kind = {'min': 'mean', 'max': 'mean', 'mean': 'mean', 'std': 'mean', 'range': 'mean', 'mean_abs': 'mean',
                    'b': 'mean'}
            _, _, _, _, _, _, b = SM().get_tensor_stats(stat_id, kind)
        else:
            b = torch.mean(torch.abs(tensor - tensor.mean())).cpu().numpy()
        return self.alpha_laplace[self.num_bits] * b

    def get_alpha_gaus(self, tensor, tag, stat_id=None):
        if tag == 'activation' and len(tensor.shape) == 4:
            N = tensor.shape[1]*tensor.shape[2]*tensor.shape[3]
        else:
            N = tensor.view(-1).size()[0]
        if stat_id is not None:
            kind = {'min': 'mean', 'max': 'mean', 'mean': 'mean', 'std': 'mean', 'range': 'mean', 'mean_abs': 'mean',
                    'b': 'mean'}
            min_value, max_value, _, std, _, _, _ = SM().get_tensor_stats(stat_id, kind)
        else:
            # TODO: Average over batch
            min_value = tensor.min()
            max_value = tensor.max()

        std = ((max_value - min_value) * self.gaussian_const) / ((2 * math.log(N)) ** 0.5)
        return self.alpha_gaus[self.num_bits] * std

    def alpha2DeltaOffset(self, alpha, max_value, min_value, mean):
        max_range = max_value - min_value
        if alpha <= 0 or alpha >= max_range / 2:
            delta = max_range
        else:
            delta = 2 * alpha
            min_value = max(min_value, mean - delta / 2)

        return delta, min_value

    # Python implementation of gemmlowp quantization. Use for quick poc and experiments
    def gemmlowpClippingQuantize(self, input, tag="", stat_id=None, clip_type='laplace'):
        if stat_id is not None:
            kind = {'min': 'mean', 'max': 'mean', 'mean': 'mean', 'std': 'mean', 'range': 'mean', 'mean_abs': 'mean',
                    'b': 'mean'}
            min_value, max_value, mean, std, range, mean_abs, b = SM().get_tensor_stats(stat_id, kind)
        else:
            min_value = input.min()
            max_value = input.max()
            mean = input.mean()

        max_range = max_value - min_value

        if clip_type == 'laplace':
            alpha = self.get_alpha_laplace(input, stat_id)  # laplace clipping
        elif clip_type == 'gaus':
            alpha = self.get_alpha_gaus(input, tag, stat_id)  # gaussian clipping
        elif clip_type == 'exp':
            alpha = self.get_alpha_exp(input, stat_id)  # exponential clipping
        elif clip_type == 'mix':
            alpha_laplace = self.get_alpha_laplace(input, stat_id)  # laplace clipping
            alpha_gause = self.get_alpha_gaus(input, tag, stat_id)  # gaussian clipping
            mse_est_laplace = IntQuantizer.mse_laplace(b, alpha_laplace, self.num_bits)
            mse_est_gaus = IntQuantizer.mse_gaus(std, alpha_gause, self.num_bits)
            if mse_est_laplace < mse_est_gaus:
                alpha = alpha_laplace
            else:
                alpha = alpha_gause
        elif clip_type == 'test':
            mse_laplace, mse_laplace_est = self.__clip_and_mse_mesure(input, tag, stat_id, 'laplace', max_value, min_value, mean, std, b)
            mse_gaus, mse_gaus_est = self.__clip_and_mse_mesure(input, tag, stat_id, 'gaus', max_value, min_value, mean, std, b)
            mse_no_clip, _ = self.__clip_and_mse_mesure(input, tag, stat_id, 'no', max_value, min_value, mean, std, b)
            min_mse_id = np.argmin([mse_no_clip, mse_laplace, mse_gaus])
            clippings = ['no clipping', 'laplace', 'gaussian']

            print("%s - MSE no clipping: %f, laplace: %f, laplace est: %f, gaussian: %f, gaussian est: %f, min mse: %s, bits: %d, std: %f, b: %f" %
                  (stat_id, mse_no_clip, mse_laplace, mse_laplace_est, mse_gaus, mse_gaus_est, clippings[min_mse_id], self.num_bits, std, b))
            return input
        else:
            # no clipping
            alpha = max_range/2

        delta, min_value = self.alpha2DeltaOffset(alpha, max_value, min_value, mean)
        res = self.__gemmlowpQuantize__(input.contiguous(), delta, min_value)

        return res

    def gemmlowpQuantize(self, tensor, min_value=None, max_value=None, mean=None):
        # TODO: Average over batch
        if min_value is None:
            min_value = tensor.detach().min()
        if max_value is None:
            max_value = tensor.detach().max()

        range = max_value - min_value
        return self.__gemmlowpQuantize__(tensor, range, min_value)

    @staticmethod
    def mse_laplace(b, alpha, num_bits):
        return 2 * (b ** 2) * np.exp(-alpha / b) + ((alpha ** 2) / (3 * 2 ** (2 * num_bits)))

    @staticmethod
    def mse_gaus(sigma, alpha, num_bits):
        clipping_err = (sigma ** 2 + (alpha ** 2)) * (1 - math.erf(alpha / (sigma * np.sqrt(2.0)))) - \
                       np.sqrt(2.0 / np.pi) * alpha * sigma * (np.e ** ((-1) * (0.5 * (alpha ** 2)) / sigma ** 2))
        quant_err = (alpha ** 2) / (3 * (2 ** (2 * num_bits)))
        return clipping_err + quant_err

    def __clip_and_mse_mesure(self, tensor, tag, stat_id, clip_type, max_value, min_value, mean, std, b):
        if clip_type == 'laplace':
            alpha = self.get_alpha_laplace(tensor, stat_id)  # laplace clipping
            mse_est = IntQuantizer.mse_laplace(b, alpha, self.num_bits)
        elif clip_type == 'gaus':
            alpha = self.get_alpha_gaus(tensor, tag, stat_id)  # gaussian clipping
            mse_est = IntQuantizer.mse_gaus(std, alpha, self.num_bits)
        else: # no clipping
            alpha = (max_value - min_value)/2
            mse_est = -1

        delta, min_value = self.alpha2DeltaOffset(alpha, max_value, min_value, mean)
        res = self.__gemmlowpQuantize__(tensor.contiguous(), delta, min_value)
        mse = torch.mean((tensor - res)**2)
        del res
        return mse, mse_est


    def __gemmlowpQuantize__(self, tensor, delta, offset):
        if self.stochastic:
            # Generate noise for stochastic rounding
            noise = tensor.new(tensor.shape).uniform_(-0.5, 0.5)
        else:
            noise = torch.cuda.FloatTensor(tensor.shape).fill_(0)

        # if enforce_true_zero and zero in range
        preserve_zero = self.enforce_true_zero and (offset + delta) > 0 and offset < 0
        return int_quantization.float2gemmlowp(tensor.contiguous(), delta, offset, self.num_bits, self.int_exp, preserve_zero, noise)

def int_quantizer(qtype, quant_params):
    if len(qtype) > len('int'):
        size = int(qtype[len('int'):])
    else:
        size = 32

    return IntQuantizer(size, quant_params)
