import argparse
import os
import os.path as osp
import numpy
import mmcv
import torch
import cv2
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmpose.datasets.pipelines import Compose
from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.core import wrap_fp16_model
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils import ExtendedDictAction
from mmpose.utils.deployment.onnxruntime_backend import ModelONNXRuntime
from mmpose.utils.deployment.openvino_backend import Model as ModelOpenVINO
from mmpose.core.evaluation import (aggregate_results, get_group_preds,
                                    get_multi_stage_outputs)

import inspect


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('model', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--device', default='CPU', help='OpenVINO inference device')
    parser.add_argument(
        '--eval',
        default='mAP',
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--update_config',
        nargs='+',
        action=ExtendedDictAction,
        dest='cfg_options',
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--update_config model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1

class LoadImage:
    """A simple pipeline to load image."""

    def __init__(self, color_type='color', channel_order='rgb'):
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the img_or_path.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img_or_path'], str):
            results['image_file'] = results['img_or_path']
            img = mmcv.imread(results['img_or_path'], self.color_type,
                              self.channel_order)
        elif isinstance(results['img_or_path'], np.ndarray):
            results['image_file'] = ''
            if self.color_type == 'color' and self.channel_order == 'rgb':
                img = cv2.cvtColor(results['img_or_path'], cv2.COLOR_BGR2RGB)
        else:
            raise TypeError('"img_or_path" must be a numpy array or a str or '
                            'a pathlib.Path object')

        results['img'] = img
        return results



def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    args.work_dir = osp.join('./work_dirs',
                             osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    # build the model and load checkpoint
    if args.model.endswith('.onnx'):
        model = ModelONNXRuntime(args.model, cfg)
    elif args.model.endswith('.xml'):
        model = ModelOpenVINO(args.model, device=args.device, cfg=cfg)

        # cfg.data.test.pipeline[0]['channel_order'] = 'bgr'
        # bura = [v for v in cfg.data.test.pipeline if v['type'] == 'BottomUpResizeAlign'][0]
        # print(bura)
        # normalize = [v for v in bura['transforms'] if v['type'] == 'NormalizeTensor'][0]
        # print(normalize)
        # print('read mean/std from config')
        # normalize['mean'] = [0, 0, 0]
        # normalize['std'] = [1. / 255., 1. / 255., 1. / 255.]
        # print(bura)

    else:
        raise ValueError('Unknown model type.')
    channel_order = cfg.test_pipeline[0].get('channel_order', 'rgb')
    test_pipeline = [LoadImage(channel_order=channel_order)
                     ] + cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        num_gpus=1,
        dist=False,
        shuffle=False,
        drop_last=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    outputs = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        img_metas = data['img_metas'].data[0]
        center, scale = img_metas[0]["center"],img_metas[0]["scale"]
        w = scale[0] * 200 / 1.25
        h = scale[1] * 200 / 1.25
        x1 = int(center[0] - w/2) if int(center[0] - w/2)>0 else 0
        x2 = int(center[0] + w/2) if int(center[0] + w/2)>0 else 0
        y1 = int(center[1] - h/2) if int(center[1] - h/2)>0 else 0
        y2 = int(center[1] + h/2) if int(center[1] + h/2)>0 else 0
        im_data = cv2.imread(img_metas[0]['image_file'])
        crop = im_data[y1:y2, x1:x2, :]
        crop=cv2.resize(crop, (64,64))
        crop = crop.transpose(2,0,1)
        inference_result = model(crop)["3851"]
        res = model.pt_model.keypoint_head.decode(img_metas, inference_result)
        boxes = []
        boxes.append(numpy.concatenate([img_metas[0]["center"],img_metas[0]["scale"],numpy.array([0,1,1])]))
        result = {}
        outputs.append(res)

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()

    eval_config = cfg.get('eval_config', {})
    eval_config = merge_configs(eval_config, dict(metric=args.eval))

    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)

    print(dataset.evaluate(outputs, args.work_dir, **eval_config))


if __name__ == '__main__':
    main()
