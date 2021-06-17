# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
import os.path as osp
import string
from collections import OrderedDict

import numpy as np
from openvino.inference_engine import IECore

from mmpose.models import build_posenet


class Model:
    def __init__(self, model_path, ie=None, device='CPU', cfg=None):
        self.logger = logging.getLogger()
        self.logger.info('Reading network from IR...')

        self.ie = IECore() if ie is None else ie
        bin_path = osp.splitext(model_path)[0] + '.bin'
        self.net = self.ie.read_network(model_path, bin_path)

        self.device = None
        self.exec_net = None
        self.to(device)

        self.pt_model = None
        if cfg is not None:
            self.pt_model = self.pt_model = build_posenet(cfg.model)

    def to(self, device):
        if self.device != device:
            self.device = device
            self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=1)
        return self

    def unify_inputs(self, inputs):
        if not isinstance(inputs, dict):
            inputs_dict = {next(iter(self.net.input_info)): inputs}
        else:
            inputs_dict = inputs
        return inputs_dict

    def reshape(self, inputs=None, input_shapes=None):
        assert (inputs is None) != (input_shapes is None)
        if input_shapes is None:
            input_shapes = {name: data.shape for name, data in inputs.items()}
        reshape_needed = False
        for input_name, input_shape in input_shapes.items():
            blob_shape = self.net.input_info[input_name].input_data.shape
            if not np.array_equal(input_shape, blob_shape):
                reshape_needed = True
                break
        print(input_shapes)
        print(inputs)
        if reshape_needed:
            self.logger.info(f'reshape net to {input_shapes}')
            print(f'reshape net to {input_shapes}')
            self.net.reshape(input_shapes)
            self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=1)

    def get(self, outputs, name):
        try:
            key = self.net.get_ov_name_for_tensor(name)
            assert key in outputs, f'"{key}" is not a valid output identifier'
        except KeyError:
            if name not in outputs:
                raise KeyError(f'Failed to identify output "{name}"')
            key = name
        return outputs[key]

    def preprocess(self, inputs):
        return inputs

    def postprocess(self, outputs):
        return outputs

    def __call__(self, inputs):
        inputs = self.unify_inputs(inputs)
        inputs = self.preprocess(inputs)
        # self.reshape(inputs=inputs)
        outputs = self.exec_net.infer(inputs)
        outputs = self.postprocess(outputs)
        return outputs

    # FIXME.
    def show(self, data, result, dataset=None, score_thr=0.3, wait_time=0):
        if self.pt_model is not None:
            self.pt_model.show_result(
                data, result, show=True, score_thr=score_thr, wait_time=wait_time)
