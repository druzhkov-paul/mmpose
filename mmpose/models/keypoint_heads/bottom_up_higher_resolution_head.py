import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_upsample_layer, constant_init,
                      normal_init)

from mmpose.core.evaluation import (aggregate_results, get_group_preds,
                                    get_multi_stage_outputs)
from mmpose.core.post_processing.decoder_ae import AssociativeEmbeddingDecoder
from mmpose.models.builder import build_loss
from ..backbones.resnet import BasicBlock, Bottleneck
from ..builder import HEADS


@HEADS.register_module()
class BottomUpHigherResolutionHead(nn.Module):
    """Bottom-up head for Higher Resolution.

    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints
        tag_per_joint (bool): If tag_per_joint is True,
            the dimension of tags equals to num_joints,
            else the dimension of tags is 1. Default: True
        extra:
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        cat_output (list[bool]): Option to concat outputs.
        with_ae_loss (list[bool]): Option to use ae loss.
        loss_keypoint (dict): Config for loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 tag_per_joint=True,
                 extra=None,
                 upsample_type='deconv',
                 num_deconv_layers=1,
                 num_deconv_filters=(32, ),
                 num_deconv_kernels=(4, ),
                 num_basic_blocks=4,
                 block_type='Basic',
                 cat_output=None,
                 with_ae_loss=None,
                 loss_keypoint=None,
                 test_cfg={}):
        super().__init__()

        self.loss = build_loss(loss_keypoint)
        dim_tag = num_joints if tag_per_joint else 1

        self.num_deconvs = num_deconv_layers
        self.cat_output = cat_output

        final_layer_output_channels = []

        if with_ae_loss[0]:
            out_channels = num_joints + dim_tag
        else:
            out_channels = num_joints

        final_layer_output_channels.append(out_channels)
        for i in range(num_deconv_layers):
            if with_ae_loss[i + 1]:
                out_channels = num_joints + dim_tag
            else:
                out_channels = num_joints
            final_layer_output_channels.append(out_channels)

        deconv_layer_output_channels = []
        for i in range(num_deconv_layers):
            if with_ae_loss[i]:
                out_channels = num_joints + dim_tag
            else:
                out_channels = num_joints
            deconv_layer_output_channels.append(out_channels)

        self.final_layers = self._make_final_layers(
            in_channels, final_layer_output_channels, extra, num_deconv_layers,
            num_deconv_filters)

        assert block_type in {'Basic', 'Bottleneck'}
        block = BasicBlock if block_type == 'Basic' else Bottleneck

        self.deconv_layers = self._make_deconv_layers(
            in_channels, deconv_layer_output_channels, num_deconv_layers,
            num_deconv_filters, num_deconv_kernels, num_basic_blocks,
            cat_output, upsample_type, block)

        self.test_cfg = test_cfg

        self.decoder = AssociativeEmbeddingDecoder(
            num_joints=self.test_cfg['num_joints'],
            max_num_people=30,
            detection_threshold=0.1,
            use_detection_val=True,
            ignore_too_much=False,
            tag_threshold=1.0,
            adjust=self.test_cfg['adjust'],
            refine=self.test_cfg['refine'],
            dist_reweight=self.test_cfg.get('dist_reweight', False),
            delta=self.test_cfg.get('delta', 0.0)
        )
        nms_kernel = self.test_cfg['nms_kernel']
        self.kpts_nms_pool = torch.nn.MaxPool2d(nms_kernel, 1, (nms_kernel - 1) // 2)

    @staticmethod
    def _make_final_layers(in_channels, final_layer_output_channels, extra,
                           num_deconv_layers, num_deconv_filters):
        """Make final layers."""
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            else:
                padding = 0
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        final_layers = []
        final_layers.append(
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=final_layer_output_channels[0],
                kernel_size=kernel_size,
                stride=1,
                padding=padding))

        for i in range(num_deconv_layers):
            in_channels = num_deconv_filters[i]
            final_layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=final_layer_output_channels[i + 1],
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding))

        return nn.ModuleList(final_layers)

    def _make_deconv_layers(self, in_channels, deconv_layer_output_channels,
                            num_deconv_layers, num_deconv_filters,
                            num_deconv_kernels, num_basic_blocks, cat_output,
                            upsample_type='deconv', block=BasicBlock):
        """Make deconv layers."""
        deconv_layers = []
        for i in range(num_deconv_layers):
            if cat_output[i]:
                in_channels += deconv_layer_output_channels[i]

            planes = num_deconv_filters[i]
            deconv_kernel, padding, output_padding = \
                self._get_deconv_cfg(num_deconv_kernels[i])

            layers = []
            assert upsample_type in {'deconv', 'bilinear'}
            if upsample_type == 'deconv':
                layers.append(
                    nn.Sequential(
                        build_upsample_layer(
                            dict(type='deconv'),
                            in_channels=in_channels,
                            out_channels=planes,
                            kernel_size=deconv_kernel,
                            stride=2,
                            padding=padding,
                            output_padding=output_padding,
                            bias=False),
                        nn.BatchNorm2d(planes, momentum=0.1),
                        nn.ReLU(inplace=True)))
            else:
                layers.append(
                    nn.Sequential(
                        build_upsample_layer(
                            dict(type='bilinear'),
                            scale_factor=2),
                        nn.Conv2d(in_channels, planes, 1),
                        nn.BatchNorm2d(planes, momentum=0.1),
                        nn.ReLU(inplace=True)))

            for _ in range(num_basic_blocks):
                layers.append(nn.Sequential(block(planes, planes), ))
            deconv_layers.append(nn.Sequential(*layers))
            in_channels = planes

        return nn.ModuleList(deconv_layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def get_loss(self, output, targets, masks, joints):
        """Calculate bottom-up keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            num_outputs: O
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            targets(List(torch.Tensor[NxKxHxW])): Multi-scale target heatmaps.
            masks(List(torch.Tensor[NxHxW])): Masks of multi-scale target
                                              heatmaps
            joints(List(torch.Tensor[NxMxKx2])): Joints of multi-scale target
                                                 heatmaps for ae loss
        """

        losses = dict()

        heatmaps_losses, push_losses, pull_losses = self.loss(
            output, targets, masks, joints)

        for idx in range(len(targets)):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                if 'heatmap_loss' not in losses:
                    losses['heatmap_loss'] = heatmaps_loss
                else:
                    losses['heatmap_loss'] += heatmaps_loss
            if push_losses[idx] is not None:
                push_loss = push_losses[idx].mean(dim=0)
                if 'push_loss' not in losses:
                    losses['push_loss'] = push_loss
                else:
                    losses['push_loss'] += push_loss
            if pull_losses[idx] is not None:
                pull_loss = pull_losses[idx].mean(dim=0)
                if 'pull_loss' not in losses:
                    losses['pull_loss'] = pull_loss
                else:
                    losses['pull_loss'] += pull_loss

        return losses

    def aggregate_augm_results(self, outputs, outputs_flipped=(), scale_factors=(1, ),
                  base_size=None, flip_index=None, use_udp=False):
        num_outputs = len(outputs)
        num_outputs_flipped = len(outputs_flipped)
        num_outputs_total = max(num_outputs, num_outputs_flipped)
        assert num_outputs_total > 0
        assert num_outputs == num_outputs_flipped or min(num_outputs, num_outputs_flipped) == 0
        assert num_outputs_total == len(scale_factors)

        aggregated_heatmaps = None
        tags_list = []
        for idx, s in enumerate(scale_factors):
            o = outputs[idx] if outputs else None
            of = outputs_flipped[idx] if outputs_flipped else None
            _, heatmaps, tags = get_multi_stage_outputs(
                o,
                of,
                self.test_cfg['num_joints'],
                self.test_cfg['with_heatmaps'],
                self.test_cfg['with_ae'],
                self.test_cfg['tag_per_joint'],
                flip_index,
                self.test_cfg['project2image'],
                base_size,
                align_corners=use_udp,
                flip_offset=self.test_cfg.get('flip_offset', 0))

            aggregated_heatmaps, tags_list = aggregate_results(
                s,
                aggregated_heatmaps,
                tags_list,
                heatmaps,
                tags,
                scale_factors,
                self.test_cfg['project2image'],
                self.test_cfg.get('flip_test', True),
                align_corners=use_udp)

        # average heatmaps of different scales
        aggregated_heatmaps = aggregated_heatmaps / float(num_outputs_total)
        aggregated_tags = torch.cat(tags_list, dim=4)

        # perform grouping
        torch.nn.functional.relu(aggregated_heatmaps, inplace=True)
        maxm = self.kpts_nms_pool(aggregated_heatmaps)
        maxm = torch.eq(maxm, aggregated_heatmaps).float()
        aggregated_heatmaps *= 2 * maxm - 1

        return aggregated_heatmaps, aggregated_tags

    def get_poses(self, aggregated_heatmaps, aggregated_tags, center=(0, 0), scale=(1, 1), use_udp=False):
        heatmaps = to_numpy(aggregated_heatmaps)
        tags = to_numpy(aggregated_tags)[..., 0]
        grouped_joints, scores = self.decoder(heatmaps, tags, heatmaps)
        poses = get_group_preds(
            [grouped_joints],
            center,
            scale,
            [aggregated_heatmaps.shape[3], aggregated_heatmaps.shape[2]],
            use_udp=use_udp)
        return poses, scores

    def forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x = x[0]

        final_outputs = []
        y = self.final_layers[0](x)
        final_outputs.append(y)

        for i in range(self.num_deconvs):
            if self.cat_output[i]:
                x = torch.cat((x, y), 1)

            x = self.deconv_layers[i](x)
            y = self.final_layers[i + 1](x)
            final_outputs.append(y)

        return final_outputs

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for _, m in self.final_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)


def to_numpy(x, dtype=np.float32):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    assert isinstance(x, np.ndarray)
    x = x.astype(dtype)
    return x
