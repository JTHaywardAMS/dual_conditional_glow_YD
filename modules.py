import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_feature, compute_same_pad, compute_same_pad2


def gaussian_p(mean, logs, x):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = logs ** 2
    """
    c = math.log(2 * math.pi)
    return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + c)


def gaussian_likelihood(mean, logs, x):
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])


def gaussian_sample(mean, logs, temperature=1):
    # Sample from Gaussian with temperature
    z = torch.normal(mean, torch.exp(logs) * temperature)

    return z


def squeeze2d(input, factor):
    if factor == 1:
        return input

    B, C, H, W = input.size()

    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x


def unsqueeze2d(input, factor):
    if factor == 1:
        return input

    factor2 = factor ** 2

    B, C, H, W = input.size()

    assert C % (factor2) == 0, "C module factor squared is not 0"

    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)

    return x


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.
    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = - torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3],
                              keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited = True

    def _center(self, input, reverse=False):
        if reverse:
            return input - self.bias
        else:
            return input + self.bias

    def _scale(self, input, logdet=None, reverse=False):

        if reverse:
            input = input * torch.exp(-self.logs)
        else:
            input = input * torch.exp(self.logs)

        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply by number of pixels
            """
            b, c, h, w = input.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return input, logdet

    def forward(self, input,  logdet=None, reverse=False):
        self._check_input_dim(input)


        if not self.inited:
            self.initialize_parameters(input)

        if reverse:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        else:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)

        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))


class LinearZeros(nn.Module):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.logscale_factor = logscale_factor

        self.logs = nn.Parameter(torch.zeros(out_channels))

    def forward(self, input):

        output = self.linear(input)
        print("logs", self.logs)
        print("exp", torch.exp(self.logs * self.logscale_factor))
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding="same", do_actnorm=True, weight_std=0.05):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=(not do_actnorm))

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, input):

        x = self.conv(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        # print(x.size())
        return x

class Conv2d_extra(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding="same", do_actnorm=True, weight_std=0.05, logscale_factor=3):
        super().__init__()

        if padding == "same":
            padding_a = compute_same_pad((3, 3), (1,1))
            padding_b = compute_same_pad((1, 1), (1,1))
        elif padding == "valid":
            padding = 0

        # self.in_channels = in_channels
        # # print("in", in_channels)
        # self.hidden_channels = hidden_channels
        # # print("hidden", hidden_channels)
        # self.out_channels = out_channels
        # # print("hidden", in_channels)
        # self.project_ycond_in = LinearZeros(num_classes, hidden_channels)
        # self.project_ycond_hidden = LinearZeros(num_classes, hidden_channels)
        # self.project_ycond_out = LinearZeros(num_classes, out_channels)

        self.cond_fc1 = nn.Linear(num_classes, int(in_channels * 96 * 96 / (in_channels * in_channels)))
        self.conv1 = nn.Conv2d(in_channels +in_channels , hidden_channels, stride,
                              padding_a, bias=(not do_actnorm))
        self.cond_conv1 = nn.Conv2d(in_channels,in_channels, kernel_size, stride,
                              padding_a, bias=(not do_actnorm))

        self.cond_fc2 = nn.Linear(num_classes, int(in_channels * 96 * 96 / (in_channels * in_channels)))
        self.conv2 = nn.Conv2d(hidden_channels + in_channels, hidden_channels,kernel_size=(1,1),  stride=stride,
                              padding=padding_b,  bias=(not do_actnorm))
        self.cond_conv2 = nn.Conv2d(in_channels, in_channels,  kernel_size=(1,1),  stride=stride,
                                   padding=padding_b,  bias=(not do_actnorm))

        self.cond_fc3 = nn.Linear(num_classes, int(in_channels * 96 * 96 / (in_channels * in_channels)))
        self.conv3 = nn.Conv2d(hidden_channels +in_channels, out_channels, kernel_size, stride,
                              padding_a)

        self.cond_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding_a)

        # init weight with std
        nn.init.normal_(self.cond_fc1.weight, 0., std=weight_std)
        nn.init.normal_(self.cond_fc2.weight, 0., std=weight_std)
        nn.init.normal_(self.cond_fc3.weight, 0., std=weight_std)

        self.conv1.weight.data.normal_(mean=0.0, std=weight_std)
        self.cond_conv1.weight.data.normal_(mean=0.0, std=weight_std)

        self.conv2.weight.data.normal_(mean=0.0, std=weight_std)
        self.cond_conv2.weight.data.normal_(mean=0.0, std=weight_std)

        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()
        self.cond_conv3.weight.data.zero_()
        self.cond_conv3.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm1 = ActNorm2d(hidden_channels)
            self.actnorm2 = ActNorm2d(hidden_channels)

        self.do_actnorm = do_actnorm

    def forward(self, input):
        x, y = input

        cond1 = self.cond_fc1(y)
        cond1 = cond1.view(x.size())
        cond1=F.relu(cond1, inplace=False)
        cond_1 = self.cond_conv1(cond1)
        x = torch.cat([x, cond_1], dim=1)
        x = self.conv1(x)

        if self.do_actnorm:
            x, _ = self.actnorm1(x)

        x=F.relu(x, inplace=False)

        cond2 = self.cond_fc2(y)
        cond2 = cond2.view(cond1.size())
        cond2 = F.relu(cond2, inplace=False)
        cond_2 = self.cond_conv2(cond2)
        x = torch.cat([x, cond_2], dim=1)
        x = self.conv2(x)

        if self.do_actnorm:
            x, _ = self.actnorm2(x)

        x=F.relu(x, inplace=False)

        cond3 = self.cond_fc3(y)
        cond3 = cond3.view(cond1.size())
        cond3 = F.relu(cond3, inplace=False)
        cond_3 = self.cond_conv3(cond3)
        x = torch.cat([x, cond_3], dim=1)
        x = self.conv3(x)

        return x * torch.exp(self.logs * self.logscale_factor)

class Conv2dZeros(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding="same", logscale_factor=3):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):

        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        self.indices = torch.arange(self.num_channels - 1, -1, -1,
                                    dtype=torch.long)
        self.indices_inverse = torch.zeros((self.num_channels),
                                           dtype=torch.long)

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        shuffle_idx = torch.randperm(self.indices.shape[0])
        self.indices = self.indices[shuffle_idx]

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):

        assert len(input.size()) == 4

        if not reverse:
            input = input[:, self.indices, :, :]
            return input
        else:
            return input[:, self.indices_inverse, :, :]


class Split2d(nn.Module):
    def __init__(self, num_channels, y_classes, sp_condition):
        super().__init__()
        self.sp_condition=sp_condition
        if sp_condition:
            self.conv = Conv2dZeros(num_channels, num_channels)
            self.cond_fc = nn.Linear(y_classes, int(num_channels * 96 * 96 / (num_channels * num_channels)) * 2)
            self.cond_conv = Conv2dZeros(num_channels // 2, num_channels // 2)
        else:
            self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def split2d_prior(self, z, condition):

        if self.sp_condition is False:
            # print('no split prior')
            h= self.conv(z)
        else:
            print('split prior')
            cond = self.cond_fc(condition)
            cond = cond.view(z.size())
            cond = self.cond_conv(cond)
            cond = F.relu(cond, inplace=False)
            z = torch.cat([z, cond], dim=1)
            h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, input, y_onehot, logdet=0., reverse=False, temperature=None):


        if reverse:
            z1 = input
            mean, logs = self.split2d_prior(z1,y_onehot)
            z2 = gaussian_sample(mean, logs, temperature)
            z = torch.cat((z1, z2), dim=1)
            return z, logdet
        else:
            z1, z2 = split_feature(input, "split")
            mean, logs = self.split2d_prior(z1, y_onehot)
            logdet = gaussian_likelihood(mean, logs, z2) + logdet
            return z1, logdet


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, y_onehot, logdet=None, reverse=False):


        if reverse:
            output = unsqueeze2d(input, self.factor)
        else:
            output = squeeze2d(input, self.factor)

        return output, logdet


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer('p', p)
            self.register_buffer('sign_s', sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """

        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet