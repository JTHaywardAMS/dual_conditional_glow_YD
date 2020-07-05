import math

import torch
import torch.nn as nn

from modules import (Conv2d,  Conv2d_extra, Conv2dZeros, ActNorm2d, InvertibleConv1x1,
                     Permute2d, LinearZeros, SqueezeLayer,
                     Split2d, gaussian_likelihood, gaussian_sample)
from utils import split_feature, uniform_binning_correction


def get_block(in_channels, out_channels, hidden_channels):
    block = nn.Sequential(Conv2d(in_channels, hidden_channels),
                          nn.ReLU(inplace=False),
                          Conv2d(hidden_channels, hidden_channels,
                                 kernel_size=(1, 1)),
                          nn.ReLU(inplace=False),
                          Conv2dZeros(hidden_channels, out_channels))
    return block

def get_block_extra(in_channels, out_channels, hidden_channels, num_classes):
    block = nn.Sequential(Conv2d_extra(in_channels, hidden_channels, out_channels, num_classes))
    return block

class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, actnorm_scale,
                 flow_permutation, flow_coupling, LU_decomposed, extra_condition, num_classes):
        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)
        self.extra_condition=extra_condition

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels,
                                             LU_decomposed=LU_decomposed)
            self.flow_permutation = \
                lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = \
                lambda z, logdet, rev: (self.shuffle(z, rev), logdet)
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = \
                lambda z, logdet, rev: (self.reverse(z, rev), logdet)

        # 3. coupling
        if flow_coupling == "additive":
            if self.extra_condition:

                self.block = get_block_extra(in_channels // 2,
                                       in_channels // 2,
                                       hidden_channels,
                                         num_classes)
            else:
                self.block = get_block(in_channels // 2,
                                   in_channels // 2,
                                   hidden_channels)
        elif flow_coupling == "affine":
            if self.extra_condition:
                self.block = get_block_extra(in_channels // 2,
                                       in_channels,
                                       hidden_channels,
                                         num_classes)
            else:
                self.block = get_block(in_channels // 2,
                                   in_channels,
                                   hidden_channels)
                print("coupling", in_channels // 2,in_channels,hidden_channels)

    def forward(self, input, y_onehot, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, y_onehot, logdet)
        else:
            return self.reverse_flow(input, y_onehot, logdet)

    def normal_flow(self, input, y_onehot, logdet):
        assert input.size(1) % 2 == 0
        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            if self.extra_condition:

                z2 = z2 + self.block((z1, y_onehot))
            else:

                z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            if self.extra_condition:

                h = self.block((z1, y_onehot))
            else:

                h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, y_onehot, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            if self.extra_condition:
                z2 = z2 - self.block((z1, y_onehot))
            else:
                z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            if self.extra_condition:

                h = self.block((z1, y_onehot))
            else:
                h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L,
                 actnorm_scale, flow_permutation, flow_coupling,
                 LU_decomposed, extra_condition, sp_condition, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        self.L = L

        H, W, C = image_shape
        print("image shape", H, W, C)

        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])
            print("after squeeze", C)

            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(in_channels=C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation=flow_permutation,
                             flow_coupling=flow_coupling,
                             LU_decomposed=LU_decomposed,
                             extra_condition=extra_condition,
                             num_classes=num_classes))
                self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < L - 1:
                print("split2d num channels", C)
                self.layers.append(Split2d(num_channels=C, y_classes=num_classes, sp_condition=sp_condition))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2
                print("after split", C)

    def forward(self, input, y_onehot, logdet=0., reverse=False, temperature=None):
        if reverse:
            return self.decode(input, y_onehot, temperature)
        else:

            return self.encode(input, y_onehot, logdet)

    def encode(self, z, y_onehot, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):

            z, logdet = layer(z, y_onehot, logdet, reverse=False)
        return z, logdet

    def decode(self, z, y_onehot, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, y_onehot, logdet=0, reverse=True,
                                  temperature=temperature)
            else:
                z, logdet = layer(z, y_onehot, logdet=0, reverse=True)
        return z


class Glow(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L, actnorm_scale,
                 flow_permutation, flow_coupling, LU_decomposed, y_classes,
                 learn_top, y_condition, extra_condition, sp_condition, d_condition, yd_condition):
        super().__init__()
        self.flow = FlowNet(image_shape=image_shape,
                            hidden_channels=hidden_channels,
                            K=K,
                            L=L,
                            actnorm_scale=actnorm_scale,
                            flow_permutation=flow_permutation,
                            flow_coupling=flow_coupling,
                            LU_decomposed=LU_decomposed,
                            extra_condition=extra_condition,
                            sp_condition=sp_condition,
                            num_classes=y_classes)
        self.y_classes = y_classes
        if y_condition or d_condition or yd_condition:
            self.y_condition = True
            print("conditional version", self.y_condition)
        else:
            self.y_condition=False
        self.learn_top = learn_top
        print("extra condtion", extra_condition)
        print("split prior condition", sp_condition)
        # learned prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        if self.y_condition:
            C = self.flow.output_shapes[-1][1]
            print("prior", 2*C)
            self.project_ycond = LinearZeros(y_classes, 2 * C)
            self.project_class = LinearZeros(C, y_classes)

        self.register_buffer("prior_h",
                             torch.zeros([1,
                                          self.flow.output_shapes[-1][1] * 2,
                                          self.flow.output_shapes[-1][2],
                                          self.flow.output_shapes[-1][3]]))

    def prior(self, data, y_onehot=None, batch_size=32):
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            # Hardcoded a batch size of 32 here
            h = self.prior_h.repeat(batch_size, 1, 1, 1)

        channels = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            if data is not None:
                h += yp.view(data.shape[0], channels, 1, 1)
            else:
                print("no data")
                h += yp.view(batch_size, channels, 1, 1)

        return split_feature(h, "split")

    def forward(self, x=None, y_onehot=None, z=None, temperature=None,
                reverse=False, batch_size=32):
        if reverse:
            return self.reverse_flow(z, y_onehot, temperature, batch_size)
        else:
            return self.normal_flow(x, y_onehot)

    def normal_flow(self, x, y_onehot):
        b, c, h, w = x.shape

        x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, y_onehot, logdet=logdet, reverse=False)

        mean, logs = self.prior(x, y_onehot)
        objective += gaussian_likelihood(mean, logs, z)

        if self.y_condition:
            print("size", z.mean(2).mean(2).size())
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # Full objective - converted to bits per dimension
        bpd = (-objective) / (math.log(2.) * c * h * w)

        return z, bpd, y_logits

    def reverse_flow(self, z, y_onehot, temperature, batch_size):
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z, y_onehot, batch_size)
                z = gaussian_sample(mean, logs, temperature)
            x = self.flow(z, y_onehot, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True