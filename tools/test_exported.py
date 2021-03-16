import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.core import wrap_fp16_model
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils.deployment.onnxruntime_backend import ModelONNXRuntime
from mmpose.utils.deployment.openvino_backend import Model as ModelOpenVINO
from mmpose.core.evaluation import (aggregate_results, get_group_preds,
                                    get_multi_stage_outputs)


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


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    args.work_dir = osp.join('./work_dirs',
                             osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

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

    # build the model and load checkpoint
    if args.model.endswith('.onnx'):
        model = ModelONNXRuntime(args.model, cfg)
    elif args.model.endswith('.xml'):
        model = ModelOpenVINO(args.model, device=args.device, cfg=cfg)
    else:
        raise ValueError('Unknown model type.')

    outputs = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        img_metas = data['img_metas'].data[0][0]
        im_data = img_metas['aug_data'][0].cpu().numpy()
        result = model(im_data)

        heatmaps = result['heatmaps']
        tags = result['embeddings']

        # grouped, scores = model.pt_model.decoder(heatmaps, tags, heatmaps)
        center = img_metas['center']
        scale = img_metas['scale']
        poses, scores = model.pt_model.keypoint_head.get_poses(heatmaps, tags, center, scale, model.pt_model.use_udp)
        # results = get_group_preds(
        #     [grouped],
        #     center,
        #     scale, [heatmaps.shape[3],
        #             heatmaps.shape[2]],
        #     use_udp=model.pt_model.use_udp)
        image_path = []
        image_path.extend(img_metas['image_file'])
        result['preds'] = poses
        result['scores'] = scores
        result['image_paths'] = image_path
        result['output_heatmap'] = None

        outputs.append(result)

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
