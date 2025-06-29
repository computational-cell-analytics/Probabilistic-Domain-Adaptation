from functools import partial

import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal, Independent

from torch_em.loss.dice import DiceLossWithLogits


class Planar(nn.Module):
    """
    PyTorch implementation of planar flows as presented in "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow parameters.
    """

    def __init__(self):

        super(Planar, self).__init__()

        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        """ Derivative of tanh """

        return 1 - self.h(x) ** 2

    def forward(self, zk, u, w, b):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """

        zk = zk.unsqueeze(-1)
        bs = u.shape[0]
        total = zk.shape[0]
        latent_dim = zk.shape[1]
        sample_size = total // bs

        if total != bs:
            u = u.unsqueeze(1).repeat(1, sample_size, 1, 1)
            w = w.unsqueeze(1).repeat(1, sample_size, 1, 1)
            b = b.unsqueeze(1).repeat(1, sample_size, 1, 1)
            u = u.reshape(bs*sample_size, latent_dim, 1)
            w = w.reshape(bs*sample_size, 1, latent_dim)
            b = b.reshape(bs*sample_size, 1, 1)

        # reparameterize u such that the flow becomes invertible (see appendix paper)
        uw = torch.bmm(w, u)
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat
        wzb = torch.bmm(w, zk) + b
        # magnitude_u = torch.sum(torch.abs(u_hat))/u_hat.shape[0]
        # magnitude_wzb = torch.sum(torch.abs(wzb))/wzb.shape[0]
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)

        # compute logdetJ
        psi = w * self.der_h(wzb)
        # jacobian = 1 + torch.bmm(torch.bmm(u_hat, self.der_h(torch.bmm(w, zk) + b)), w)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        return log_det_jacobian, z


class Radial(nn.Module):
    """
    PyTorch implementation of planar flows as presented in "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow parameters.
    """

    def __init__(self, shape):

        super(Radial, self).__init__()

        self.d_cpu = torch.prod(torch.tensor(shape))
        self.register_buffer('d', self.d_cpu)
        # lim = 1.0 / np.prod(shape)

    def forward(self, zk, z0, alpha, beta):  # TODO: fix
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """

        beta = torch.log(1 + torch.exp(beta)) - torch.abs(alpha)
        zk = zk.unsqueeze(-1)
        bs = alpha.shape[0]
        total = zk.shape[0]
        latent_dim = zk.shape[1]
        sample_size = total // bs
        if total != bs:
            alpha = alpha.unsqueeze(1).repeat(1, sample_size, 1, 1)
            beta = beta.unsqueeze(1).repeat(1, sample_size, 1, 1)
            z0 = z0.unsqueeze(1).repeat(1, sample_size, 1, 1)
            alpha = alpha.reshape(bs*sample_size, 1, 1)
            beta = beta.reshape(bs*sample_size, 1, 1)
            z0 = z0.reshape(bs*sample_size, latent_dim, 1)

        dz = zk - z0

        r = torch.norm(dz, dim=list(range(1, z0.dim() - 1))).unsqueeze(-1)
        h_arr = beta / (torch.abs(alpha) + r)
        h_arr_ = - beta * r / (torch.abs(alpha) + r) ** 2
        zk = zk + h_arr * dz
        log_det_jacobian = (self.d - 1) * torch.log(1 + h_arr) + torch.log(1 + h_arr + h_arr_)
        # log_det always positive?
        return log_det_jacobian, zk.squeeze(-1)


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


def conv3d(in_channels, out_channels, kernel_size, bias, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
            'bcl' -> batchnorm + conv + LeakyReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, \
                f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'  # noqa
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))

        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))

        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super().__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super().__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels

        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module(
            'SingleConv1',
            SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups, padding=padding)
        )
        # conv2
        self.add_module(
            'SingleConv2',
            SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups, padding=padding)
        )


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        apply_pooling=True,
        pool_kernel_size=2,
        pool_type='max',
        basic_module=DoubleConv,
        conv_layer_order='gcr',
        num_groups=8,
        padding=1
    ):
        super().__init__()

        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(
            in_channels, out_channels,
            encoder=True,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


def create_encoders(
    in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, pool_kernel_size
):
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(
                in_channels, out_feature_num,
                apply_pooling=False,  # skip pooling in the first encoder
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                padding=conv_padding
            )

        elif i == (len(f_maps)-1):
            encoder = Encoder(
                f_maps[i - 1], out_feature_num,
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                pool_kernel_size=pool_kernel_size,
                pool_type='avg',
                padding=conv_padding
            )
        else:
            # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to
            # make the data isotropic after 1-2 pooling operations
            encoder = Encoder(
                f_maps[i - 1], out_feature_num,
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                pool_kernel_size=pool_kernel_size,
                padding=conv_padding
            )

        encoders.append(encoder)

    return nn.ModuleList(encoders)


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True

    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2, 2)):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        upsample = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor, padding=1
        )
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation)
    followed by a basic module (DoubleConv or ExtResNetBlock).
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (boole): should the input be upsampled
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        scale_factor=(2, 2, 2),
        basic_module=DoubleConv,
        conv_layer_order='gcr',
        num_groups=8,
        mode='nearest',
        padding=1,
        upsample=True
    ):
        super(Decoder, self).__init__()

        if upsample:
            if basic_module == DoubleConv:
                # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
                self.upsampling = InterpolateUpsampling(mode=mode)
                # concat joining
                self.joining = partial(self._joining, concat=True)
            else:
                # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
                self.upsampling = TransposeConvUpsampling(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=conv_kernel_size,
                    scale_factor=scale_factor
                )
                # sum joining
                self.joining = partial(self._joining, concat=False)
                # adapt the number of in_channels for the ExtResNetBlock
                in_channels = out_channels
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        self.basic_module = basic_module(
            in_channels, out_channels,
            encoder=False,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding
        )

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample):
    # create decoder path consisting of the Decoder modules.
    # The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        _upsample = True
        if i == 0:
            # upsampling can be skipped only for the 1st decoder, afterwards it should always be present
            _upsample = upsample

        decoder = Decoder(
            in_feature_num, out_feature_num,
            basic_module=basic_module,
            conv_layer_order=layer_order,
            conv_kernel_size=conv_kernel_size,
            num_groups=num_groups,
            padding=conv_padding,
            upsample=_upsample
        )
        decoders.append(decoder)

    return nn.ModuleList(decoders)


class AbstractAxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(
        self,
        in_channels,
        f_maps,
        basic_module,
        conv_kernel_size,
        conv_padding,
        layer_order,
        num_groups,
        pool_kernel_size,
        latent_dim,
        posterior=False
    ):
        super(AbstractAxisAlignedConvGaussian, self).__init__()
        self.input_channels = in_channels
        self.channel_axis = 1
        self.num_filters = f_maps
        self.latent_dim = latent_dim
        self.posterior = posterior

        if self.posterior:
            self.name = 'Posterior'
            self.encoder = create_encoders(
                self.input_channels,
                self.num_filters,
                basic_module,
                conv_kernel_size,
                conv_padding,
                layer_order,
                num_groups,
                pool_kernel_size
            )
        else:
            self.name = 'Prior'
            self.encoder = create_encoders(
                self.input_channels,
                self.num_filters,
                basic_module,
                conv_kernel_size,
                conv_padding,
                layer_order,
                num_groups,
                pool_kernel_size
            )

        self.conv_layer = nn.Conv3d(f_maps[-1], 2 * self.latent_dim, (1, 1, 1), stride=1)

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

    def forward(self, input, segm=None):

        # If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        x = input
        # encoding = self.encoder(input)
        for encoder in self.encoder:
            x = encoder(x)
        encoding = x
        self.show_enc = encoding

        # We only want the mean of the resulting hxwxd image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        encoding = torch.mean(encoding, dim=4, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return encoding.squeeze(-1).squeeze(-1).squeeze(-1), dist


class PriorNet(AbstractAxisAlignedConvGaussian):
    def __init__(
        self,
        in_channels,
        f_maps,
        basic_module,
        conv_kernel_size,
        conv_padding,
        layer_order,
        num_groups,
        pool_kernel_size,
        latent_dim,
        posterior=False
    ):
        super().__init__(
            in_channels=in_channels,
            f_maps=f_maps,
            basic_module=basic_module,
            conv_kernel_size=conv_kernel_size,
            conv_padding=conv_padding,
            layer_order=layer_order,
            num_groups=num_groups,
            pool_kernel_size=pool_kernel_size,
            latent_dim=latent_dim,
            posterior=posterior
        )


class PosteriorNet(AbstractAxisAlignedConvGaussian):
    def __init__(
        self,
        in_channels,
        f_maps,
        basic_module,
        conv_kernel_size,
        conv_padding,
        layer_order,
        num_groups,
        pool_kernel_size,
        latent_dim,
        posterior=True
    ):
        super().__init__(
            in_channels=in_channels,
            f_maps=f_maps,
            basic_module=basic_module,
            conv_kernel_size=conv_kernel_size,
            conv_padding=conv_padding,
            layer_order=layer_order,
            num_groups=num_groups,
            pool_kernel_size=pool_kernel_size,
            latent_dim=latent_dim,
            posterior=posterior
        )


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.normal_(m.weight, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


class PosteriorNetWithNormalizingFlow(nn.Module):
    def __init__(
        self,
        in_channels,
        f_maps,
        basic_module,
        conv_kernel_size,
        conv_padding,
        layer_order,
        num_groups,
        pool_kernel_size,
        latent_dim,
        num_flow_steps,
        flow_type,
        posterior=True
    ):
        super().__init__()

        self.num_flow_steps = num_flow_steps
        self.latent_dim = latent_dim

        self.samples = 1  # samples

        self.flow_type = flow_type
        self.base_posterior = PosteriorNet(
            in_channels,
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            pool_kernel_size,
            latent_dim
        )

        nF_oP = num_flow_steps * latent_dim

        if flow_type == Radial:
            # Amortized flow parameters
            self.amor_z0 = nn.Sequential(
                nn.Linear(f_maps[-1], nF_oP),
                nn.ReLU(),
                nn.Linear(nF_oP, nF_oP),
                nn.BatchNorm1d(nF_oP)
            )
            self.amor_alpha = nn.Sequential(
                nn.Linear(f_maps[-1], num_flow_steps),
                nn.ReLU(),
                nn.Linear(num_flow_steps, num_flow_steps),
                nn.BatchNorm1d(num_flow_steps)
            )
            self.amor_beta = nn.Sequential(
                nn.Linear(f_maps[-1], num_flow_steps),
                nn.ReLU(),
                nn.Linear(num_flow_steps, num_flow_steps),
                nn.BatchNorm1d(num_flow_steps)
            )

            self.amor_z0.apply(init_weights)
            self.amor_alpha.apply(init_weights)
            self.amor_beta.apply(init_weights)
            # Normalizing flow layers
            for k in range(num_flow_steps):
                flow_k = flow_type(shape=self.latent_dim)  # .to(device)
                self.add_module('flow_' + str(k), flow_k)

        elif flow_type == Planar:
            # Amortized flow parameters
            self.amor_u = nn.Sequential(
                nn.Linear(f_maps[-1], nF_oP),
                nn.ReLU(),
                nn.Linear(nF_oP, nF_oP),
                nn.BatchNorm1d(nF_oP)
            )
            self.amor_w = nn.Sequential(
                nn.Linear(f_maps[-1], nF_oP),
                nn.ReLU(),
                nn.Linear(nF_oP, nF_oP),
                nn.BatchNorm1d(nF_oP)
            )
            self.amor_b = nn.Sequential(
                nn.Linear(f_maps[-1], num_flow_steps),
                nn.ReLU(),
                nn.Linear(num_flow_steps, num_flow_steps),
                nn.BatchNorm1d(num_flow_steps)
            )

            self.amor_u.apply(init_weights)
            self.amor_w.apply(init_weights)
            self.amor_b.apply(init_weights)

            for k in range(num_flow_steps):
                flow_k = flow_type()  # .to(device)
                self.add_module('flow_' + str(k), flow_k)

    def forward(self, input, segm=None):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0 [sum_k log |det dz_k/dz_k-1| ].
        """
        batch_size = input.shape[0]
        h, z0_density = self.base_posterior(input, segm)

        z0 = z0_density.rsample((self.samples,)).reshape(self.samples * batch_size, self.latent_dim)
        z = [z0]

        if self.flow_type == Radial:
            self.z0 = self.amor_z0(h).view(batch_size, self.num_flow_steps, self.latent_dim, 1)
            self.alpha = self.amor_alpha(h).view(batch_size, self.num_flow_steps, 1, 1)
            self.beta = self.amor_beta(h).view(batch_size, self.num_flow_steps, 1, 1)

            log_det_j, z = self.radial_flow(z)
            return log_det_j, z[0], z[-1], z0_density

        elif self.flow_type == Planar:
            self.u = self.amor_u(h).view(batch_size, self.num_flow_steps, self.latent_dim, 1)
            self.w = self.amor_w(h).view(batch_size, self.num_flow_steps, 1, self.latent_dim)
            self.b = self.amor_b(h).view(batch_size, self.num_flow_steps, 1, 1)

            log_det_j, z = self.planar_flow(z)
            return log_det_j, z[0], z[-1], z0_density

    def radial_flow(self, z):
        log_det_j = 0.

        for k in range(self.num_flow_steps):
            flow_k = getattr(self, 'flow_' + str(k))

            log_det_jacobian, z_k = flow_k(z[k], self.z0[:, k, :, :], self.alpha[:, k, :, :], self.beta[:, k, :, :])
            z.append(z_k)
            log_det_j += log_det_jacobian

        return log_det_j.mean(), z

    def planar_flow(self, z):

        log_det_j = 0.

        for k in range(self.num_flow_steps):
            flow_k = getattr(self, 'flow_' + str(k))
            log_det_jacobian, z_k = flow_k(z[k], self.u[:, k, :, :], self.w[:, k, :, :], self.b[:, k, :, :])
            z.append(z_k)
            log_det_j += log_det_jacobian

        return log_det_j, z


def init_weights_orthogonal_normal(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)


class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(
        self,
        num_filters,
        latent_dim,
        num_output_channels,
        num_classes,
        no_convs_fcomb,
        initializers,
        use_tile=True
    ):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels  # output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3, 4]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []
            # Decoder of N x a 1x1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv3d(self.num_filters[0]+self.latent_dim, self.num_filters[0], kernel_size=(1, 1, 1)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv3d(self.num_filters[0], self.num_filters[0], kernel_size=(1, 1, 1)))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)
            self.last_layer = nn.Conv3d(self.num_filters[0], self.num_classes, kernel_size=(1, 1, 1))

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
        ).to(device='cuda')
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_size x latent_dim and feature_map is batch_size x no_channelsxDxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxWxD. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])
            z = torch.unsqueeze(z, 4)
            z = self.tile(z, 4, feature_map.shape[self.spatial_axes[2]])

            # Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)

            output = self.layers(feature_map)
            return self.last_layer(output)


# https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
# Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing
def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L


def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio)  # step is in [0,1]

    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop:
            L[int(i+c*period)] = 1.0 / (1.0 + np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L


#  function  = 1 − cos(a), where a scans from 0 to pi/2
def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio)  # step is in [0,1]

    # transform into [0, pi] for plots:
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop:
            L[int(i+c*period)] = 0.5 - 0.5 * math.cos(v * math.pi)
            v += step
            i += 1
    return L


class ELBO(nn.Module):
    """Combination of BCE and KL Divergence losses"""
    def __init__(
        self,
        start=1.0,
        stop=10.0,
        n_epoch=300000,
        n_cycle=4,
        ratio=0.5,
        beta_scheduler='cosine',
        beta_magnitude=10.0,
        rl_swap=True,
        **kwargs
    ):
        super().__init__()

        if rl_swap:
            self.reconstruction_loss = DiceLossWithLogits()
        else:
            self.reconstruction_loss = nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction=None)

        if beta_scheduler == "cosine":
            self.beta_schedule = beta_magnitude * frange_cycle_cosine(start, stop, n_epoch, n_cycle, ratio)
        elif beta_scheduler == "linear":
            self.beta_schedule = beta_magnitude * frange_cycle_linear(start, stop, n_epoch, n_cycle, ratio)
        elif beta_scheduler == "sigmoid":
            self.beta_schedule = beta_magnitude * frange_cycle_sigmoid(start, stop, n_epoch, n_cycle, ratio)

    def forward(self, input, target, kl_div, iteration_num=None):
        reconstruction_loss = self.reconstruction_loss(input, target)

        if iteration_num is None:
            return - (reconstruction_loss + 1e-5 * kl_div)
        else:
            return - (reconstruction_loss + self.beta_schedule[iteration_num] * kl_div)
