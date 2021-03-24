import inspect
import math
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.nn.modules.batchnorm import _BatchNorm
from pytorchcv.models.common import round_channels, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block, SEBlock
from pytorchcv.models.efficientnet import EfficientNet
from pytorchcv.models.hrnet import HRFinalBlock


CONV_LAYERS = {'Conv': nn.Conv2d, 'Conv2d': nn.Conv2d}

def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


NORM_LAYERS = {'BN': nn.BatchNorm2d,
    'BN2d': nn.BatchNorm2d}


def infer_abbr(class_type):
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm'


def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        (str, nn.Module): The first element is the layer name consisting of
            abbreviation and postfix, e.g., bn1, gn. The second element is the
            created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        # elif issubclass(block, Bottleneck):
        #     expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class HRModule(nn.Module):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output=False,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, num_blocks, in_channels,
                        num_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_CHANNELS({len(num_channels)})'
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        """Make one branch."""
        downsample = None
        if stride != 1 or \
                self.in_channels[branch_index] != \
                num_channels[branch_index] * get_expansion(block):
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    num_channels[branch_index] * get_expansion(block),
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(
                    self.norm_cfg,
                    num_channels[branch_index] * get_expansion(block))[1])

        layers = []
        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index] * get_expansion(block),
                stride=stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        self.in_channels[branch_index] = \
            num_channels[branch_index] * get_expansion(block)
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index] * get_expansion(block),
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class CompactEfficientNet(EfficientNet):
    def __init__(self, depth_factor=1.0, width_factor=1.0, dropout_rate=0.2,
                 num_classes=1000, **kwargs):
        init_block_channels = 32
        layers = [1, 2, 2, 3, 3, 4, 1]
        downsample = [1, 1, 1, 1, 0, 1, 0]
        channels_per_layers = [16, 24, 40, 80, 112, 192, 320]
        expansion_factors_per_layers = [1, 6, 6, 6, 6, 6, 6]
        kernel_sizes_per_layers = [3, 3, 5, 3, 5, 5, 3]
        strides_per_stage = [1, 2, 2, 2, 1, 2, 1]
        final_block_channels = 1280

        layers = [int(math.ceil(li * depth_factor)) for li in layers]
        channels_per_layers = [round_channels(ci * width_factor) for ci in channels_per_layers]

        from functools import reduce
        channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                        zip(channels_per_layers, layers, downsample), [])
        kernel_sizes = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                            zip(kernel_sizes_per_layers, layers, downsample), [])
        expansion_factors = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                                zip(expansion_factors_per_layers, layers, downsample), [])
        strides_per_stage = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                                zip(strides_per_stage, layers, downsample), [])
        strides_per_stage = [si[0] for si in strides_per_stage]

        init_block_channels = round_channels(init_block_channels * width_factor)

        if width_factor > 1.0:
            assert (int(final_block_channels * width_factor) == round_channels(final_block_channels * width_factor))
            final_block_channels = round_channels(final_block_channels * width_factor)

        self.init_block_channels = init_block_channels
        self.channels = channels

        super().__init__(
            channels=channels,
            init_block_channels=init_block_channels,
            final_block_channels=final_block_channels,
            kernel_sizes=kernel_sizes,
            strides_per_stage=strides_per_stage,
            expansion_factors=expansion_factors,
            dropout_rate=0.2,
            tf_mode=False,
            bn_eps=1e-5,
            in_size=192)


class EfficientHRNet(CompactEfficientNet):
    def __init__(self, depth_factor=1.0, width_factor=1.0, dropout_rate=0.2,
                 width_per_branch=(32, 64, 128, 256), blocks_per_stage=(1, 4, 3),
                 out_indices=None, num_classes=1000, **kwargs):
        super().__init__(depth_factor, width_factor, dropout_rate, num_classes, **kwargs)

        self.out_indices = out_indices

        assert len(width_per_branch) == len(out_indices)
        out_channels = [self.init_block_channels, ] + [c[-1] for c in self.channels]
        width_per_branch = list(width_per_branch)
        out_channels = [out_channels[i] for i in out_indices]
        self.transition = self._make_transition_layer(out_channels, width_per_branch)

        self.stages = nn.ModuleList()
        for stage_idx, blocks_num in enumerate(blocks_per_stage, 2):
            self.stages.append(nn.Sequential(*[
                HRModule(stage_idx,
                    BasicBlock,
                    [2, ] * stage_idx,
                    width_per_branch[:stage_idx],
                    width_per_branch[:stage_idx],
                    # [x // 4 for x in width_per_branch[:stage_idx]],
                    multiscale_output=True,
                    with_cp=False,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN')) for i in range(blocks_num)]))

        self.stages.append(HRFinalBlock(width_per_branch, [128, 256, 512, 1024]))

        self.final_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.output = nn.Linear(
            in_features=2048,
            out_features=num_classes)

        self._init_params()

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                None,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(dict(type='BN'),
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                None,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False),
                            build_norm_layer(dict(type='BN'), out_channels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)


    def forward(self, x):
        outputs = []
        y = x

        last_stage = max(self.out_indices)
        for i, stage in enumerate(self.features):
            y = stage(y)
            if i in self.out_indices:
                outputs.append(y)
            if i == last_stage:
                break

        # FIXME.
        if len(outputs) == 1:
            outputs = outputs[0]

        # print(self.transition)
        # print('-' * 30)
        # print(self.stages)

        x = list(transition(y) for transition, y in zip(self.transition, outputs))
        for stage_idx, stage in enumerate(self.stages, 2):
            # print(f'stage {stage_idx - 2}')
            # print('in')
            # for xxx in x[:stage_idx]:
            #     print(xxx.shape)
            y = stage(x[:stage_idx])
            if isinstance(y, (list, tuple)):
                x[:stage_idx] = y
            else:
                x = y
            # print('out')
            # for xxx in x[:stage_idx]:
            #     print(xxx.shape)

        # if isinstance(x, (list, tuple)):
        #     print(list(xx.shape for xx in x))
        # else:
        #     print(x.shape)

        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_efficienthrnet(version,
                     in_size,
                     bn_eps=1e-5,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join("~", ".torch", "models"),
                     **kwargs):
    if version == "bm0":
        assert (in_size == (224, 224))
        model_index = 0
        dropout_rate = 0.2
        width_per_branch=(32, 64, 128, 256)
        blocks_per_stage=(1, 4, 3)
    elif version == "bm1":
        assert (in_size == (192, 192))
        model_index = -1
        dropout_rate = 0.2
        width_per_branch=(26, 52, 103, 206)
        blocks_per_stage=(1, 3, 3)
    elif version == "bm2":
        assert (in_size == (160, 160))
        model_index = -2
        dropout_rate = 0.3
        width_per_branch = (21, 42, 83, 166)
        blocks_per_stage = (1, 2, 3)
    elif version == "bm3":
        assert (in_size == (128, 128))
        model_index = -3
        dropout_rate = 0.3
        width_per_branch = (17, 34, 67, 133)
        blocks_per_stage = (1, 1, 3)
    elif version == "bm4":
        assert (in_size == (128, 128))
        model_index = -4
        dropout_rate = 0.4
        width_per_branch = (14, 27, 54, 107)
        blocks_per_stage = (1, 1, 2)
    else:
        raise ValueError("Unsupported EfficientHRNet version {}".format(version))

    out_indices = (2, 3, 4, 5)
    depth_factor = pow(1.2, model_index)
    width_factor = pow(1.1, model_index)

    net = EfficientHRNet(
        depth_factor=depth_factor,
        width_factor=width_factor,
        dropout_rate=dropout_rate,
        width_per_branch=width_per_branch,
        blocks_per_stage=blocks_per_stage,
        out_indices=out_indices,
        in_size=in_size,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def get_compact_efficientnet(version,
                             in_size,
                             bn_eps=1e-5,
                             model_name=None,
                             pretrained=False,
                             root=os.path.join("~", ".torch", "models"),
                             **kwargs):
    if version == "bm0":
        assert (in_size == (224, 224))
        model_index = 0
        dropout_rate = 0.2
    elif version == "bm1":
        assert (in_size == (195, 195))
        model_index = -1
        dropout_rate = 0.2
    elif version == "bm2":
        assert (in_size == (170, 170))
        model_index = -2
        dropout_rate = 0.3
    elif version == "bm3":
        assert (in_size == (145, 145))
        model_index = -3
        dropout_rate = 0.3
    elif version == "bm4":
        assert (in_size == (128, 128))
        model_index = -4
        dropout_rate = 0.4
    else:
        raise ValueError("Unsupported EfficientNet version {}".format(version))

    depth_factor = pow(1.2, model_index)
    width_factor = pow(1.1, model_index)

    net = CompactEfficientNet(
        depth_factor=depth_factor,
        width_factor=width_factor,
        dropout_rate=dropout_rate,
        in_size=in_size,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def efficienthrnet_bm0(in_size=(224, 224), **kwargs):
    return get_efficienthrnet(version="bm0", in_size=in_size, model_name="efficienthrnet_bm0", **kwargs)

def efficienthrnet_bm1(in_size=(192, 192), **kwargs):
    return get_efficienthrnet(version="bm1", in_size=in_size, model_name="efficienthrnet_bm1", **kwargs)

def efficienthrnet_bm2(in_size=(160, 160), **kwargs):
    return get_efficienthrnet(version="bm2", in_size=in_size, model_name="efficienthrnet_bm2", **kwargs)

def efficienthrnet_bm3(in_size=(128, 128), **kwargs):
    return get_efficienthrnet(version="bm3", in_size=in_size, model_name="efficienthrnet_bm3", **kwargs)

def efficienthrnet_bm4(in_size=(128, 128), **kwargs):
    return get_efficienthrnet(version="bm4", in_size=in_size, model_name="efficienthrnet_bm4", **kwargs)

def efficientnet_bm1(in_size=(195, 195), **kwargs):
    return get_compact_efficientnet(version="bm1", in_size=in_size, model_name="efficientnet_bm1", **kwargs)

def efficientnet_bm2(in_size=(170, 170), **kwargs):
    return get_compact_efficientnet(version="bm2", in_size=in_size, model_name="efficientnet_bm2", **kwargs)

def efficientnet_bm3(in_size=(145, 145), **kwargs):
    return get_compact_efficientnet(version="bm3", in_size=in_size, model_name="efficientnet_bm3", **kwargs)

def efficientnet_bm4(in_size=(128, 128), **kwargs):
    return get_compact_efficientnet(version="bm4", in_size=in_size, model_name="efficientnet_bm4", **kwargs)
