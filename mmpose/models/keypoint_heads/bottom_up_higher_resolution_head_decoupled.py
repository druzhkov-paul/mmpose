import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_upsample_layer, constant_init,
                      normal_init)

from ..backbones.resnet import BasicBlock, Bottleneck
from ..builder import HEADS
from .bottom_up_higher_resolution_head import BottomUpHigherResolutionHead
from mmpose.core.evaluation import aggregate_results, get_group_preds


@HEADS.register_module()
class BottomUpHigherResolutionHeadDecoupled(BottomUpHigherResolutionHead):
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
        super().__init__(in_channels,
            num_joints,
            tag_per_joint,
            extra,
            upsample_type,
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels,
            num_basic_blocks,
            block_type,
            cat_output,
            with_ae_loss,
            loss_keypoint,
            test_cfg)

        dim_tag = num_joints if tag_per_joint else 1

        final_layer_output_channels = []
        for i in range(num_deconv_layers + 1):
            final_layer_output_channels.append((num_joints, dim_tag if with_ae_loss[i] else 0))

        self.final_hmap_layers, self.final_tag_layers = self._make_final_layers_x(
            in_channels, final_layer_output_channels, extra, num_deconv_layers,
            num_deconv_filters)

        # Remove redundant weights. Those will be shared with the ones in BottomUpHigherResolutionHead.
        self.final_hmap_weights_shape = []
        for x_, y_ in zip(self.final_layers, self.final_hmap_layers):
            self.final_hmap_weights_shape.append(y_.weight.shape)
            del y_.weight
            del y_.bias

        self.final_tag_weights_shape = []
        for x_, y_ in zip(self.final_layers, self.final_tag_layers):
            if y_ is None:
                self.final_tag_weights_shape.append(None)
                continue
            self.final_tag_weights_shape.append(y_.weight.shape)
            del y_.weight
            del y_.bias

    def share_weights(self):
        for x, y, shape in zip(self.final_layers, self.final_hmap_layers, self.final_hmap_weights_shape):
            y.weight = x.weight[:shape[0], :shape[1]]
            y.bias = x.bias[:shape[0]]
        for x, y, shape in zip(self.final_layers, self.final_tag_layers, self.final_tag_weights_shape):
            if y is None:
                continue
            y.weight = x.weight[-shape[0]:, -shape[1]:]
            y.bias = x.bias[-shape[0]:]

    @staticmethod
    def _make_final_layers_x(in_channels, final_layer_output_channels, extra,
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

        final_heatmap_layers = []
        final_tag_layers = []
        for i in range(num_deconv_layers + 1):
            in_channels = num_deconv_filters[i - 1] if i > 0 else in_channels
            out_channels = final_layer_output_channels[i][0]
            final_heatmap_layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding) if out_channels > 0 else None
            )
            out_channels = final_layer_output_channels[i][1]
            final_tag_layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding) if out_channels > 0 else None
            )
        return nn.ModuleList(final_heatmap_layers), nn.ModuleList(final_tag_layers)

    def forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x = x[0]

        self.share_weights()

        final_outputs = []
        y = (
            self.final_hmap_layers[0](x) if self.final_hmap_layers[0] is not None else None,
            self.final_tag_layers[0](x) if self.final_tag_layers[0] is not None else None,
        )
        final_outputs.append(y)

        for i in range(self.num_deconvs):
            y = [z for z in y if z is not None]
            y = torch.cat(y, dim=1)
            if self.cat_output[i]:
                x = torch.cat((x, y), 1)

            x = self.deconv_layers[i](x)
            y = (
                self.final_hmap_layers[i + 1](x) if self.final_hmap_layers[i + 1] is not None else None,
                self.final_tag_layers[i + 1](x) if self.final_tag_layers[i + 1] is not None else None,
            )
            final_outputs.append(y)

        return final_outputs

    @staticmethod
    def get_multi_stage_outputs(outputs,
                                outputs_flip,
                                num_joints,
                                with_heatmaps,
                                with_ae,
                                tag_per_joint=True,
                                flip_index=None,
                                project2image=True,
                                size_projected=None,
                                align_corners=False,
                                flip_offset=0):
        heatmaps_avg = 0
        num_heatmaps = 0
        heatmaps = []
        tags = []

        flip_test = outputs_flip is not None

        # # aggregate heatmaps from different stages
        # for i, output in enumerate(outputs):
        #     if i != len(outputs) - 1:
        #         output = torch.nn.functional.interpolate(
        #             output,
        #             size=(outputs[-1].size(2), outputs[-1].size(3)),
        #             mode='bilinear',
        #             align_corners=align_corners)

        #     # staring index of the associative embeddings
        #     offset_feat = num_joints if with_heatmaps[i] else 0

        #     if with_heatmaps[i]:
        #         heatmaps_avg += output[:, :num_joints]
        #         num_heatmaps += 1

        #     if with_ae[i]:
        #         tags.append(output[:, offset_feat:])

        target_size = outputs[-1][0].shape
        # aggregate heatmaps from different stages
        for i, output in enumerate(outputs):
            # starting index of the associative embeddings
            offset_feat = num_joints if with_heatmaps[i] else 0

            if with_heatmaps[i]:
                heatmap = output[0]
                heatmaps_avg += torch.nn.functional.interpolate(
                    heatmap,
                    size=(target_size[2], target_size[3]),
                    mode='bilinear',
                    align_corners=align_corners) if heatmap.shape != target_size else heatmap
                num_heatmaps += 1

            if with_ae[i]:
                tag = output[1]
                tags.append(
                    torch.nn.functional.interpolate(
                    tag,
                    size=(target_size[2], target_size[3]),
                    mode='bilinear',
                    align_corners=align_corners) if tag.shape != target_size else tag
                )

        if num_heatmaps > 0:
            heatmaps.append(heatmaps_avg / num_heatmaps)

        if flip_test and flip_index:
            # perform flip testing
            heatmaps_avg = 0
            num_heatmaps = 0

            for i, output in enumerate(outputs_flip):
                if i != len(outputs_flip) - 1:
                    output = torch.nn.functional.interpolate(
                        output,
                        size=(outputs_flip[-1].size(2), outputs_flip[-1].size(3)),
                        mode='bilinear',
                        align_corners=align_corners)
                output = torch.flip(output, [3])
                outputs.append(output)

                offset_feat = num_joints if with_heatmaps[i] else 0

                if with_heatmaps[i]:
                    o = output[:, :num_joints][:, flip_index, :, :]
                    if flip_offset > 0:
                        heatmaps_avg += torch.nn.functional.pad(o[..., :-flip_offset], (flip_offset, 0))
                    elif flip_offset < 0:
                        heatmaps_avg += torch.nn.functional.pad(o[..., -flip_offset:], (0, -flip_offset))
                    else:
                        heatmaps_avg += o
                    num_heatmaps += 1

                if with_ae[i]:
                    tags.append(output[:, offset_feat:])
                    if tag_per_joint:
                        t = tags[-1][:, flip_index, :, :]
                        if flip_offset > 0:
                            tags[-1] = torch.nn.functional.pad(t[..., :-flip_offset], (flip_offset, 0))
                        elif flip_offset < 0:
                            tags[-1] = torch.nn.functional.pad(t[..., -flip_offset:], (0, -flip_offset))
                        else:
                            tags[-1] = t

            heatmaps.append(heatmaps_avg / num_heatmaps)

        if project2image and size_projected:
            heatmaps = [
                torch.nn.functional.interpolate(
                    hms,
                    size=(size_projected[1], size_projected[0]),
                    mode='bilinear',
                    align_corners=align_corners) for hms in heatmaps
            ]

            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(size_projected[1], size_projected[0]),
                    mode='bilinear',
                    align_corners=align_corners) for tms in tags
            ]

        return outputs, heatmaps, tags

    def aggregate_augm_results(self, outputs, outputs_flipped=(), scale_factors=(1, ),
                  base_size=(0, 0), flip_index=None, use_udp=False):
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
            _, heatmaps, tags = self.get_multi_stage_outputs(
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