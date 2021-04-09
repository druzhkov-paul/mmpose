import logging
import tempfile
import types

import torch.nn as nn
from mmcv.runner import load_checkpoint
from pytorchcv.model_provider import get_model
from torch.nn.modules.batchnorm import _BatchNorm

from ..registry import BACKBONES


@BACKBONES.register_module()
class EHRNet(nn.Module):
    def __init__(self, version='', norm_eval=False):
        super().__init__()
        self.module = get_model(version)
        self.norm_eval = norm_eval

        del self.module.output
        del self.module.final_pool
        del self.module.stages[-1]

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            from mmpose.utils import get_root_logger
            logger = get_root_logger()
            # load_checkpoint(self.module, pretrained, strict=False, map_location='cpu', logger=logger)
            load_checkpoint(self.module, pretrained, strict=False, logger=logger)

    def forward(self, x):
        outputs = []
        y = x

        last_stage = max(self.module.out_indices)
        for i, stage in enumerate(self.module.features):
            y = stage(y)
            if i in self.module.out_indices:
                outputs.append(y)
            if i == last_stage:
                break

        # FIXME.
        if len(outputs) == 1:
            outputs = outputs[0]

        x = list(transition(y) for transition, y in zip(self.module.transition, outputs))
        for stage_idx, stage in enumerate(self.module.stages, 2):
            y = stage(x[:stage_idx])
            if isinstance(y, (list, tuple)):
                x[:stage_idx] = y
            else:
                x = y

        # if isinstance(x, (list, tuple)):
        #     print(len(x), ':', ', '.join(f'{a.shape}' for a in x))
        # else:
        #     print(x.shape)
        return x

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
