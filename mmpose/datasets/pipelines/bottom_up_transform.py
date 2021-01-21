import math

import cv2
import mmcv
import numpy as np
try:
    import albumentations
except ImportError:
    albumentations = None

from mmpose.core.post_processing import get_affine_transform
from mmpose.datasets.registry import PIPELINES
from .shared_transform import Compose


def get_warp_matrix(theta, roi_center, size_dst, roi_size):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        roi_center (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        roi_size (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        matrix (np.ndarray): A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / roi_size[0]
    scale_y = size_dst[1] / roi_size[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-roi_center[0] * math.cos(theta) +
                              roi_center[1] * math.sin(theta) +
                              0.5 * roi_size[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-roi_center[0] * math.sin(theta) -
                              roi_center[1] * math.cos(theta) +
                              0.5 * roi_size[1])
    return matrix


def warp_affine_joints(joints, mat):
    """Apply affine transformation defined by the transform matrix on the
    joints.

    Args:
        joints (np.ndarray[..., 2]): Origin coordinate of joints.
        mat (np.ndarray[3, 2]): The affine matrix.

    Returns:
        matrix (np.ndarray[..., 2]): Result coordinate of joints.
    """
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return np.dot(
        np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
        mat.T).reshape(shape)


def _ceil_to_multiples_of(x, base=64):
    """Transform x to the integral multiple of the base."""
    return int(np.ceil(x / base)) * base


def _get_multi_scale_size(image,
                          input_size,
                          current_scale,
                          min_scale,
                          use_udp=False,
                          size_divisor=64):
    """Get the size for multi-scale training.

    Args:
        image: Input image.
        input_size (int): Size of the image input.
        current_scale (float): Scale factor.
        min_scale (float): Minimal scale.
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        - (w_resized, h_resized) (tuple(int)): resized width/height
        - center (np.ndarray): image center
        - scale (np.ndarray): scales wrt width/height
    """
    h, w, _ = image.shape

    # calculate the size for min_scale
    min_input_size = _ceil_to_multiples_of(min_scale * input_size, size_divisor)
    if w < h:
        w_resized = int(min_input_size * current_scale / min_scale)
        h_resized = int(
            _ceil_to_multiples_of(min_input_size / w * h, size_divisor) * current_scale /
            min_scale)
        if use_udp:
            scale_w = w - 1.0
            scale_h = (h_resized - 1.0) / (w_resized - 1.0) * (w - 1.0)
        else:
            scale_w = w / 200.0
            scale_h = h_resized / w_resized * w / 200.0
    else:
        h_resized = int(min_input_size * current_scale / min_scale)
        w_resized = int(
            _ceil_to_multiples_of(min_input_size / h * w, size_divisor) * current_scale /
            min_scale)
        if use_udp:
            scale_h = h - 1.0
            scale_w = (w_resized - 1.0) / (h_resized - 1.0) * (h - 1.0)
        else:
            scale_h = h / 200.0
            scale_w = w_resized / h_resized * h / 200.0
    if use_udp:
        center = (scale_w / 2.0, scale_h / 2.0)
    else:
        center = np.array([round(w / 2.0), round(h / 2.0)])
    return (w_resized, h_resized), center, np.array([scale_w, scale_h])


def _resize_align_multi_scale(image, input_size, current_scale, min_scale, size_divisor):
    """Resize the images for multi-scale training.

    Args:
        image: Input image
        input_size (int): Size of the image input
        current_scale (float): Current scale
        min_scale (float): Minimal scale

    Returns:
        tuple: A tuple containing image info.

        - image_resized (tuple): size of resize image
        - center (np.ndarray): center of image
        - scale (np.ndarray): scale
    """
    size_resized, center, scale = _get_multi_scale_size(
        image, input_size, current_scale, min_scale, False, size_divisor)

    trans = get_affine_transform(center, scale, 0, size_resized)
    image_resized = cv2.warpAffine(image, trans, size_resized)

    return image_resized, center, scale


def _resize_align_multi_scale_udp(image, input_size, current_scale, min_scale, size_divisor):
    """Resize the images for multi-scale training.

    Args:
        image: Input image
        input_size (int): Size of the image input
        current_scale (float): Current scale
        min_scale (float): Minimal scale

    Returns:
        tuple: A tuple containing image info.

        - image_resized (np.ndarray): resized image
        - center (np.ndarray): center of image
        - scale (np.ndarray): scale
    """
    size_resized, _, _ = _get_multi_scale_size(image, input_size,
                                               current_scale, min_scale, True, size_divisor)

    _, center, scale = _get_multi_scale_size(image, input_size, min_scale,
                                             min_scale, True, size_divisor)

    trans = get_warp_matrix(
        theta=0,
        roi_center=np.array(scale, dtype=np.float) / 2,
        size_dst=np.array(size_resized, dtype=np.float) - 1.0,
        roi_size=np.array(scale, dtype=np.float))
    # print(f'scale = {scale}')
    # print(f'size_resized = {size_resized}')
    # print(trans)
    # print(cv2.invertAffineTransform(trans))
    # print((image.shape[1] - 1, image.shape[0] - 1), '->', (trans @ np.array([image.shape[1] - 1, image.shape[0] - 1, 1], dtype=np.float)[:, None])[:, 0])
    image_resized = cv2.warpAffine(
        image.copy(), trans, size_resized, flags=cv2.INTER_LINEAR)

    return image_resized, center, scale


class HeatmapGeneratorCustom:
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res / 64
        # TODO. make sigma keypoint and scale dependent.
        self.sigma = sigma
        self.x = np.arange(0, output_res, 1, np.float32)[None, :]
        self.y = np.arange(0, output_res, 1, np.float32)[:, None]

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    # if x < 0 or y < 0 or \
                    #    x >= self.output_res or y >= self.output_res:
                    #     continue

                    s = -0.5 / (sigma ** 2)
                    kx = np.exp((self.x - x) ** 2 * s)
                    ky = np.exp((self.y - y) ** 2 * s)
                    g = kx * ky

                    hms[idx, ...] = np.maximum(hms[idx, ...], g)
        return hms


class HeatmapGenerator:
    """Generate heatmaps for bottom-up models.

    Args:
        num_joints (int): Number of keypoints
        output_res (int): Size of feature map
        sigma (int): Sigma of the heatmaps.
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, output_res, num_joints, sigma=-1, use_udp=False):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        self.use_udp = use_udp
        if use_udp:
            self.x = np.arange(0, size, 1, np.float32)
            self.y = self.x[:, None]
        else:
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]
            x0, y0 = 3 * sigma + 1, 3 * sigma + 1
            self.g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def __call__(self, joints):
        """Generate heatmaps."""
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    if self.use_udp:
                        x0 = 3 * sigma + 1 + pt[0] - x
                        y0 = 3 * sigma + 1 + pt[1] - y
                        g = np.exp(-((self.x - x0)**2 + (self.y - y0)**2) /
                                   (2 * sigma**2))
                    else:
                        g = self.g

                    ul = int(np.round(x - 3 * sigma -
                                      1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma +
                                      2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb,
                        cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], g[a:b,
                                                                      c:d])
        return hms


class JointsEncoder:
    """Encodes the visible joints into (coordinates, score); The coordinate of
    one joint and its score are of `int` type.

    (idx * output_res**2 + y * output_res + x, 1) or (0, 0).

    Args:
        max_num_people(int): Max number of people in an image
        num_joints(int): Number of keypoints
        output_res(int): Size of feature map
        tag_per_joint(bool):  Option to use one tag map per joint.
    """

    def __init__(self, max_num_people, num_joints, output_res, tag_per_joint):
        self.max_num_people = max_num_people
        self.num_joints = num_joints
        self.output_res = output_res
        self.tag_per_joint = tag_per_joint

    def __call__(self, joints):
        """
        Note:
            number of people in image: N
            number of keypoints: K
            max number of people in an image: M

        Args:
            joints (np.ndarray[NxKx3])

        Returns:
            visible_kpts (np.ndarray[MxKx2]).
        """
        visible_kpts = np.zeros((self.max_num_people, self.num_joints, 2),
                                dtype=np.float32)
        output_res = self.output_res
        for i in range(len(joints)):
            tot = 0
            for idx, pt in enumerate(joints[i]):
                x, y = int(pt[0]), int(pt[1])
                if (pt[2] > 0 and 0 <= y < self.output_res
                        and 0 <= x < self.output_res):
                    if self.tag_per_joint:
                        visible_kpts[i][tot] = \
                            (idx * output_res**2 + y * output_res + x, 1)
                    else:
                        visible_kpts[i][tot] = (y * output_res + x, 1)
                    tot += 1
        return visible_kpts


@PIPELINES.register_module()
class BottomUpRandomFlip:
    """Data augmentation with random image flip for bottom-up.

    Args:
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        image, mask, joints = results['img'], results['mask'], results[
            'joints']
        self.flip_index = results['ann_info']['flip_index']
        self.output_size = results['ann_info']['heatmap_size']

        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        if np.random.random() < self.flip_prob:
            image = image[:, ::-1] - np.zeros_like(image)
            for i, _output_size in enumerate(self.output_size):
                mask[i] = mask[i][:, ::-1]
                joints[i] = joints[i][:, self.flip_index]
                joints[i][:, :, 0] = _output_size - joints[i][:, :, 0] - 1
        results['img'], results['mask'], results[
            'joints'] = image, mask, joints
        return results


@PIPELINES.register_module()
class BottomUpRandomAffine:
    """Data augmentation with random scaling & rotating.

    Args:
        rot_factor (int): Rotating to [-rotation_factor, rotation_factor]
        scale_factor (float): Scaling to [1-scale_factor, 1+scale_factor]
        scale_type: wrt ``long`` or ``short`` length of the image.
        trans_factor: Translation factor.
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self,
                 rot_factor,
                 scale_factor,
                 scale_type,
                 trans_factor,
                 use_udp=False):
        self.max_rotation = rot_factor
        self.min_scale = scale_factor[0]
        self.max_scale = scale_factor[1]
        self.scale_type = scale_type
        self.trans_factor = trans_factor
        self.use_udp = use_udp

    @staticmethod
    def _get_affine_matrix(center, scale, res, rot=0):
        """Generate transformation matrix."""
        h = scale
        t = np.zeros((3, 3), dtype=np.float32)
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if rot != 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3), dtype=np.float32)
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1] / 2
            t_mat[1, 2] = -res[0] / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        image, mask, joints = results['img'], results['mask'], results[
            'joints']

        self.input_size = results['ann_info']['image_size']
        self.output_size = results['ann_info']['heatmap_size']

        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size), (len(mask),
                                                    len(self.output_size),
                                                    self.output_size)

        # print(results['ann_info'])
        # print(image.shape)
        # print(self.output_size)
        # print('-' * 30)

        height, width = image.shape[:2]
        if self.use_udp:
            center = np.array(((width - 1.0) / 2, (height - 1.0) / 2))
        else:
            center = np.array((width / 2, height / 2))
        if self.scale_type == 'long':
            scale = max(height, width) / 1.0
        elif self.scale_type == 'short':
            scale = min(height, width) / 1.0
        else:
            raise ValueError('Unknown scale type: {}'.format(self.scale_type))
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) \
            + self.min_scale
        # aug_scale = 1.0
        scale *= aug_scale
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation
        # aug_rot = 0.0

        if self.trans_factor > 0:
            dx = np.random.randint(-self.trans_factor * scale / 200.0,
                                   self.trans_factor * scale / 200.0)
            dy = np.random.randint(-self.trans_factor * scale / 200.0,
                                   self.trans_factor * scale / 200.0)

            center[0] += dx
            center[1] += dy
        if self.use_udp:
            for i, _output_size in enumerate(self.output_size):
                trans = get_warp_matrix(
                    theta=aug_rot,
                    roi_center=center,
                    size_dst=np.array(
                        (_output_size, _output_size), dtype=np.float) - 1.0,
                    roi_size=np.array((scale, scale), dtype=np.float))
                # print(f'output size: {_output_size}')
                # print(trans)
                # print(cv2.invertAffineTransform(trans))
                mask[i] = cv2.warpAffine(
                    (mask[i] * 255).astype(np.uint8),
                    trans, (int(_output_size), int(_output_size)),
                    flags=cv2.INTER_LINEAR) / 255
                mask[i] = (mask[i] > 0.5).astype(np.float32)
                joints[i][:, :, 0:2] = \
                    warp_affine_joints(joints[i][:, :, 0:2].copy(), trans)
                if results['ann_info']['scale_aware_sigma']:
                    joints[i][:, :, 3] = joints[i][:, :, 3] / aug_scale
            mat_input = get_warp_matrix(
                theta=aug_rot,
                roi_center=center,
                size_dst=np.array(
                    (self.input_size, self.input_size), dtype=np.float) - 1.0,
                roi_size=np.array((scale, scale), dtype=np.float))
            # print(f'input size: {self.input_size}')
            # print(f'scale: {scale}')
            # print(mat_input)
            # print(cv2.invertAffineTransform(mat_input))
            # cv2.imshow('orig', image)
            image = cv2.warpAffine(
                image,
                mat_input, (int(self.input_size), int(self.input_size)),
                flags=cv2.INTER_LINEAR)
            # cv2.imshow('warped', image)
            # cv2.waitKey(0)
        else:
            for i, _output_size in enumerate(self.output_size):
                mat_output = self._get_affine_matrix(center, scale,
                                                     (_output_size,
                                                      _output_size),
                                                     aug_rot)[:2]
                mask[i] = cv2.warpAffine(
                    (mask[i] * 255).astype(np.uint8), mat_output,
                    (_output_size, _output_size)) / 255
                mask[i] = (mask[i] > 0.5).astype(np.float32)

                joints[i][:, :, 0:2] = \
                    warp_affine_joints(joints[i][:, :, 0:2], mat_output)
                if results['ann_info']['scale_aware_sigma']:
                    joints[i][:, :, 3] = joints[i][:, :, 3] / aug_scale
            mat_input = self._get_affine_matrix(center, scale,
                                                (self.input_size,
                                                 self.input_size), aug_rot)[:2]
            image = cv2.warpAffine(image, mat_input,
                                   (self.input_size, self.input_size))

        results['img'], results['mask'], results[
            'joints'] = image, mask, joints

        return results


@PIPELINES.register_module()
class Albu:

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 scales=(4, 2),
                 kp_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        """
        Adds custom transformations from Albumentations lib.
        Please, visit `https://albumentations.readthedocs.io`
        to get more information.

        transforms (list): list of albu transformations
        bbox_params (dict): bbox_params for albumentation `Compose`
        keymap (dict): contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): whether to skip the image
                                      if no ann left after aug
        """
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms
        self.filter_lost_elements = False
        self.kp_filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno
        self.scales = scales

        # A simple workaround to remove masks without boxes
        if bbox_params is not None:
            if 'label_fields' in bbox_params and bbox_params.get('filter_lost_elements', False):
                self.filter_lost_elements = True
                self.origin_label_fields = bbox_params['label_fields']
                bbox_params['label_fields'] = ['idx_mapper']
            if 'filter_lost_elements' in bbox_params:
                del bbox_params['filter_lost_elements']

        if kp_params is not None:
            if 'label_fields' in kp_params and kp_params.get('filter_lost_elements', False):
                self.kp_filter_lost_elements = True
                self.origin_label_fields = kp_params['label_fields']
                kp_params['label_fields'] = ['kp_idx_mapper']
            if 'filter_lost_elements' in kp_params:
                del kp_params['filter_lost_elements']

        self.bbox_params = self.albu_builder(bbox_params) if bbox_params else None
        self.kp_params = self.albu_builder(kp_params) if kp_params else None
        self.aug = albumentations.Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params,
                           keypoint_params=self.kp_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes',
                'gt_keypoints': 'keypoints'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

        print(self)

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        Inherits some of `build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                'type must be a str or valid type, but got {}'.format(
                    type(obj_type)))

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """
        Dictionary mapper.
        Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        # print('# of masks before transforms: ', results['masks'].shape)

        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        if 'keypoints' in results:
            assert len(results['keypoints']) == len(self.scales)
            results['keypoints'] = np.concatenate(results['keypoints'], axis=0)
            keypoints_per_instance = results['keypoints'].shape[1]
            # print(results['keypoints'].shape)
            results['keypoints_visibility'] = results['keypoints'][:, :, 2]
            results['keypoints'] = results['keypoints'][:, :, :2].reshape(-1, 2)
            # print(results['keypoints'].shape)
            results['keypoints'] = results['keypoints'].tolist()
            keypoints_num = len(results['keypoints'])
            # print(keypoints_num)
            if self.kp_filter_lost_elements:
                results['kp_idx_mapper'] = np.arange(keypoints_num)

        if 'masks' in results:
            masks = results['masks']
            for i, m in enumerate(masks):
                masks[i] = m.astype(np.uint8)
        # if 'masks' in results:
            # assert len(results['masks']) == len(self.scales)
            # print(len(results['masks']))
            # print(results['masks'][0].shape)
            # for x, y in zip(*results['masks']):
            #     assert x.shape == y.shape
            # results['masks'] = list(m for m in results['masks'][0])
            # print(len(results['masks']))
            # print(results['masks'][0].shape)
            # results['masks'] = np.concatenate(results['masks'], axis=0)

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)

            # filter label_fields
            if self.filter_lost_elements:
                # print('filter lost bboxes')

                results['idx_mapper'] = np.arange(len(results['bboxes']))

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'masks' in results:
            # results['masks'] = np.asarray(results['masks'])
            # print('# of masks after transforms: ', results['masks'].shape)
            # results['masks'] = np.split(results['masks'], len(self.scales), axis=0)
            # m = np.concatenate(results['masks'])
            # results['masks'] = list(m.copy() for _ in self.scales)

            for i, s in enumerate(self.scales):
                w, h = results['masks'][i].shape
                results['masks'][i] = cv2.resize(results['masks'][i], dsize=(int(w / s), int(h / s)), interpolation=cv2.INTER_LINEAR) > 0.5

        if 'keypoints' in results:
            results['keypoints'] = np.asarray(results['keypoints'], dtype=np.float32)
            # print(results['keypoints'].shape)

            if self.kp_filter_lost_elements:
                idx = np.asarray(results['kp_idx_mapper'], dtype=np.long)
                kps = np.zeros([keypoints_num, 2], dtype=np.float32)
                # print(kps.shape)
                if len(idx) > 0:
                    kps[idx, ...] = results['keypoints']
                results['keypoints'] = kps

                visibility = np.zeros(keypoints_num, dtype=np.float32)
                visibility[idx] = results['keypoints_visibility'].flatten()[idx]
                results['keypoints_visibility'] = visibility.reshape(-1, keypoints_per_instance)

                # results['kp_idx_mapper'] = np.arange(len(results['keypoints']))
                # if (not len(results['kp_idx_mapper']) and self.skip_img_without_anno):
                #     return None

            # print(results['keypoints'].shape)
            results['keypoints'] = results['keypoints'].reshape(-1, keypoints_per_instance, 2)
            results['keypoints'] = np.concatenate([results['keypoints'], results['keypoints_visibility'][:, :, None]], axis=2)
            results['keypoints'] = np.split(results['keypoints'], len(self.scales), axis=0)
            for i, s in enumerate(self.scales):
                results['keypoints'][i][:, :2] /= s

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])
            results['gt_labels'] = results['gt_labels'].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transforms={})'.format(self.transforms)
        return repr_str


@PIPELINES.register_module()
class BottomUpGenerateTarget:
    """Generate multi-scale heatmap target for bottom-up.

    Args:
        sigma (int): Sigma of heatmap Gaussian
        max_num_people (int): Maximum number of people in an image
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, sigma, max_num_people, use_udp=False):
        self.sigma = sigma
        self.max_num_people = max_num_people
        self.use_udp = use_udp

    def _generate(self, num_joints, heatmap_size):
        """Get heatmap generator and joint encoder."""
        heatmap_generator = [
            HeatmapGeneratorCustom(output_size, num_joints, self.sigma)
            for output_size in heatmap_size
        ]
        # heatmap_generator = [
        #     HeatmapGenerator(output_size, num_joints, self.sigma, self.use_udp)
        #     for output_size in heatmap_size
        # ]
        joints_encoder = [
            JointsEncoder(self.max_num_people, num_joints, output_size, True)
            for output_size in heatmap_size
        ]
        return heatmap_generator, joints_encoder

    def __call__(self, results):
        """Generate multi-scale heatmap target for bottom-up."""
        heatmap_generator, joints_encoder = \
            self._generate(results['ann_info']['num_joints'],
                           results['ann_info']['heatmap_size'])
        target_list = list()
        img, mask_list, joints_list = results['img'], results['mask'], results[
            'joints']

        for scale_id in range(results['ann_info']['num_scales']):
            target_t = heatmap_generator[scale_id](joints_list[scale_id])
            joints_t = joints_encoder[scale_id](joints_list[scale_id])

            target_list.append(target_t.astype(np.float32))
            mask_list[scale_id] = mask_list[scale_id].astype(np.float32)
            joints_list[scale_id] = joints_t.astype(np.int32)

        results['img'], results['masks'], results[
            'joints'] = img, mask_list, joints_list
        results['targets'] = target_list

        return results


@PIPELINES.register_module()
class BottomUpGetImgSize:
    """Get multi-scale image sizes for bottom-up, including base_size and
    test_scale_factor. Keep the ratio and the image is resized to
    `results['ann_info']['image_size']×current_scale`.

    Args:
        test_scale_factor (List[float]): Multi scale
        current_scale (int): default 1
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, test_scale_factor, current_scale=1, use_udp=False, size_divisor=64):
        self.test_scale_factor = test_scale_factor
        self.min_scale = min(test_scale_factor)
        self.current_scale = current_scale
        self.use_udp = use_udp
        self.size_divisor = size_divisor

    def __call__(self, results):
        """Get multi-scale image sizes for bottom-up."""
        input_size = results['ann_info']['image_size']
        img = results['img']

        h, w, _ = img.shape

        # calculate the size for min_scale
        min_input_size = _ceil_to_multiples_of(self.min_scale * input_size, self.size_divisor)
        if w < h:
            w_resized = int(min_input_size * self.current_scale /
                            self.min_scale)
            h_resized = int(
                _ceil_to_multiples_of(min_input_size / w * h, self.size_divisor) *
                self.current_scale / self.min_scale)
            if self.use_udp:
                scale_w = w - 1.0
                scale_h = (h_resized - 1.0) / (w_resized - 1.0) * (w - 1.0)
            else:
                scale_w = w / 200.0
                scale_h = h_resized / w_resized * w / 200.0
        else:
            h_resized = int(min_input_size * self.current_scale /
                            self.min_scale)
            w_resized = int(
                _ceil_to_multiples_of(min_input_size / h * w, self.size_divisor) *
                self.current_scale / self.min_scale)
            if self.use_udp:
                scale_h = h - 1.0
                scale_w = (w_resized - 1.0) / (h_resized - 1.0) * (h - 1.0)
            else:
                scale_h = h / 200.0
                scale_w = w_resized / h_resized * h / 200.0
        if self.use_udp:
            center = (scale_w / 2.0, scale_h / 2.0)
        else:
            center = np.array([round(w / 2.0), round(h / 2.0)])
        results['ann_info']['test_scale_factor'] = self.test_scale_factor
        results['ann_info']['base_size'] = (w_resized, h_resized)
        results['ann_info']['center'] = center
        results['ann_info']['scale'] = np.array([scale_w, scale_h])

        return results


@PIPELINES.register_module()
class BottomUpResizeAlign:
    """Resize multi-scale size and align transform for bottom-up.

    Args:
        transforms (List): ToTensor & Normalize
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, transforms, use_udp=False, size_divisor=64):
        self.transforms = Compose(transforms)
        self.size_divisor = size_divisor
        if use_udp:
            self._resize_align_multi_scale = _resize_align_multi_scale_udp
        else:
            self._resize_align_multi_scale = _resize_align_multi_scale

    def __call__(self, results):
        """Resize multi-scale size and align transform for bottom-up."""
        input_size = results['ann_info']['image_size']
        test_scale_factor = results['ann_info']['test_scale_factor']
        aug_data = []

        # print(f'test_scale_factors = {test_scale_factor}')
        # print('*' * 30)
        for _, s in enumerate(sorted(test_scale_factor, reverse=True)):
            _results = results.copy()
            # print('-' * 30)
            # print(f'scale = {s}')
            # print(f'net input size = {input_size}')
            # img_shape = _results['img'].shape
            # print(f'image shape = {img_shape}')
            image_resized, _, _ = self._resize_align_multi_scale(
                _results['img'], input_size, s, min(test_scale_factor), self.size_divisor)
            # print(f'resize image shape = {image_resized.shape}')
            # print('=' * 30)
            _results['img'] = image_resized
            _results = self.transforms(_results)
            transformed_img = _results['img'].unsqueeze(0)
            aug_data.append(transformed_img)

        results['ann_info']['aug_data'] = aug_data

        return results
