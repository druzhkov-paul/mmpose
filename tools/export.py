import argparse
import os.path as osp
from subprocess import run, CalledProcessError, DEVNULL

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint

from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils import ExtendedDictAction

try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError:
    raise NotImplementedError('please update mmcv to version>=1.0.4')


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def pytorch2onnx(model,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False):
    """Convert pytorch model to onnx model.

    Args:
        model (:obj:`nn.Module`): The pytorch model to be exported.
        input_shape (tuple[int]): The input tensor shape of the model.
        opset_version (int): Opset version of onnx used. Default: 11.
        show (bool): Determines whether to print the onnx model architecture.
            Default: False.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        verify (bool): Determines whether to verify the onnx model.
            Default: False.
    """
    model.cpu().eval()

    one_img = torch.randn(input_shape)

    register_extra_symbolics(opset_version)
    torch.onnx.export(
        model,
        one_img,
        output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=show,
        opset_version=opset_version,
        input_names=['image'],
        output_names=['heatmaps', 'embeddings'],
        dynamic_axes = {
            "image": {
                2: "height",
                3: "width"
            }
        }
    )

    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_result = model(one_img).detach().numpy()

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(None,
                               {net_feed_input[0]: one_img.detach().numpy()
                                })[0]
        # only compare part of results
        assert np.allclose(
            pytorch_result, onnx_result,
            atol=1.e-5), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')


def export_to_openvino(cfg, onnx_model_path, output_dir_path, input_shape=None, input_format='rgb'):
    cfg.model.pretrained = None

    output_names = 'heatmaps,embeddings'

    # Channel order is defined by LoadImage transform. By default it is 'rgb'.
    channel_order = 'rgb'
    load_image_stage = [v for v in cfg.data.test.pipeline if v['type'] == 'LoadImageFromFile']
    if len(load_image_stage) == 1:
        print('read channel order from config')
        channel_order = load_image_stage[0].get('channel_order', 'rgb')

    try:
        bura = [v for v in cfg.data.test.pipeline if v['type'] == 'BottomUpResizeAlign'][0]
        print(bura)
        normalize = [v for v in bura['transforms'] if v['type'] == 'NormalizeTensor'][0]
        print(normalize)
        print('read mean/std from config')
        # FIXME. Should those be reversed?
        mean_values = list(x * 255 for x in normalize['mean'])
        scale_values = list(x * 255 for x in normalize['std'])
    except Exception as ex:
        print(ex)
        mean_values = [0, 0, 0]
        scale_values = [1, 1, 1]

    command_line = f'mo.py --input_model="{onnx_model_path}" ' \
                   f'--mean_values="{mean_values}" ' \
                   f'--scale_values="{scale_values}" ' \
                   f'--output_dir="{output_dir_path}" ' \
                   f'--output="{output_names}"'

    assert input_format.lower() in {'bgr', 'rgb'}

    if input_shape is not None:
        command_line += f' --input_shape="{input_shape}"'
    if channel_order != input_format.lower() == 'bgr':
        command_line += ' --reverse_input_channels'

    print(command_line)

    try:
        run('mo.py -h', stdout=DEVNULL, stderr=DEVNULL, shell=True, check=True)
    except CalledProcessError:
        raise RuntimeError('OpenVINO Model Optimizer is not found or configured improperly.')

    run(command_line, shell=True, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export MMPose models to ONNX/OpenVINO')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('output_dir', help='path to directory to save exported models in')
    parser.add_argument('-ckpt', '--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--opset', type=int, default=11)
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 256, 192],
        help='input size')
    parser.add_argument('--update_config', nargs='+', action=ExtendedDictAction,
                        help='Update configuration file by parameters specified here.')
    subparsers = parser.add_subparsers(title='target', dest='target', help='target model format')
    subparsers.required = True
    parser_onnx = subparsers.add_parser('onnx', help='export to ONNX')
    parser_openvino = subparsers.add_parser('openvino', help='export to OpenVINO')
    parser_openvino.add_argument('--input_shape', nargs=2, type=int, default=None,
                                 help='input shape as a height-width pair')
    parser_openvino.add_argument('--input_format', choices=['BGR', 'RGB'], default='BGR',
                                 help='Input image format for exported model.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset == 11, 'MMPose only supports opset 11 now'

    cfg = mmcv.Config.fromfile(args.config)
    if args.update_config is not None:
        cfg.merge_from_dict(args.update_config)
    # build the model
    model = build_posenet(cfg.model)
    model = _convert_batchnorm(model)

    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    if args.checkpoint:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # convert model to onnx format
    mmcv.mkdir_or_exist(osp.abspath(args.output_dir))
    onnx_model_path = osp.join(args.output_dir, osp.splitext(osp.basename(args.config))[0] + '.onnx')
    pytorch2onnx(
        model,
        args.shape,
        opset_version=args.opset,
        show=args.show,
        output_file=onnx_model_path,
        verify=args.verify)

    if args.target == 'openvino':
        try:
            image_size = cfg['data']['test']['data_cfg']['image_size']
        except KeyError:
            image_size = cfg['image_size']
        input_shape = (1, 3, image_size, image_size)
        export_to_openvino(cfg, onnx_model_path, args.output_dir, input_shape, args.input_format)
