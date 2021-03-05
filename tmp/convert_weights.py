import torch
import argparse
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('weights_in')
    parser.add_argument('weights_out')
    parser.add_argument('--decouple', action='store_true')
    args = parser.parse_args()
    return args

def rename_weights(weights):
    dst = OrderedDict()
    for k, v in weights.items():
        key = k
        tokens = k.split('.')
        if tokens[0] == 'kp_head':
            tokens[0] = 'keypoint_head'
            if tokens[1] == 'heatmap_layers':
                tokens[1] = 'final_layers'
            if tokens[1] == 'deconv_layers':
                tokens[2] = str(int(tokens[2]) - 1)
            if tokens[1] == 'ae_layers':
                key = None
                tokens[1] = 'final_layers'
                key = '.'.join(tokens)
                dst[key] = torch.cat((dst[key], v))
                tokens = []
            key = '.'.join(tokens)

        if key:
            dst[key] = v
    return dst

def rename_weights_decoupled(weights):
    dst = OrderedDict()
    for k, v in weights.items():
        key = k
        tokens = k.split('.')
        if tokens[0] == 'kp_head':
            tokens[0] = 'keypoint_head'
            if tokens[1] == 'heatmap_layers':
                tokens[1] = 'final_hmap_layers'
            if tokens[1] == 'deconv_layers':
                tokens[2] = str(int(tokens[2]) - 1)
            if tokens[1] == 'ae_layers':
                tokens[1] = 'final_tag_layers'
            key = '.'.join(tokens)

        if key:
            dst[key] = v
    return dst

args = parse_args()
weights = torch.load(args.weights_in)
if args.decouple:
    weights['state_dict'] = rename_weights_decoupled(weights['state_dict'])
else:
    weights['state_dict'] = rename_weights(weights['state_dict'])
torch.save(weights, args.weights_out)
