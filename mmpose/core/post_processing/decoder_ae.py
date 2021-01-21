import numpy as np
from scipy.optimize import linear_sum_assignment


class Pose:
    def __init__(self, num_joints, tag_size=1):
        self.num_joints = num_joints
        self.tag_size = tag_size
        self.pose = np.zeros((num_joints, 2 + 1 + tag_size), dtype=np.float32)
        self.pose_tag = np.zeros(tag_size, dtype=np.float32)
        self.valid_points_num = 0

    def add(self, idx, joint, tag):
        self.pose[idx] = joint
        self.pose_tag = (self.pose_tag * self.valid_points_num) + tag
        self.valid_points_num += 1
        self.pose_tag /= self.valid_points_num

    @property
    def tag(self):
        if self.valid_points_num > 0:
            return self.pose_tag
        return None


class AssociativeEmbeddingDecoder:
    def __init__(self, num_joints, max_num_people, detection_threshold, use_detection_val,
                 ignore_too_much, tag_threshold,
                 adjust=True, refine=True, delta=0.0, joints_order=None):
        self.num_joints = num_joints
        self.max_num_people = max_num_people
        self.detection_threshold = detection_threshold
        self.tag_threshold = tag_threshold
        self.use_detection_val = use_detection_val
        self.ignore_too_much = ignore_too_much

        if self.num_joints == 17 and joints_order is None:
            self.joint_order = (0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16)
        else:
            self.joint_order = list(np.arange(self.num_joints))

        self.do_adjust = adjust
        self.do_refine = refine
        self.delta = delta

    def match(self, tag_k, loc_k, val_k):
        return list(map(self._match_by_tag, zip(tag_k, loc_k, val_k)))

    @staticmethod
    def _max_match(scores):
        r, c = linear_sum_assignment(scores)
        tmp = np.stack((r, c), axis=1)
        return tmp

    def _match_by_tag(self, inp):
        tag_k, loc_k, val_k = inp
        embd_size = tag_k.shape[2]
        all_joints = np.concatenate((loc_k, val_k[..., None], tag_k), -1)

        poses = []
        for idx in self.joint_order:
            tags = tag_k[idx]
            joints = all_joints[idx]
            mask = joints[:, 2] > self.detection_threshold
            tags = tags[mask]
            joints = joints[mask]

            if len(poses) == 0:
                for tag, joint in zip(tags, joints):
                    pose = Pose(self.num_joints, embd_size)
                    pose.add(idx, joint, tag)
                    poses.append(pose)
                continue

            if joints.shape[0] == 0 or (self.ignore_too_much and len(poses) == self.max_num_people):
                continue

            poses_tags = np.stack([p.tag for p in poses], axis=0)
            diff = tags[:, None] - poses_tags[None, :]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)
            if self.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]
            num_added = diff.shape[0]
            num_grouped = diff.shape[1]
            if num_added > num_grouped:
                diff_normed = np.pad(diff_normed, ((0, 0), (0, num_added - num_grouped)),
                                     mode='constant', constant_values=1e10)

            pairs = self._max_match(diff_normed)
            for row, col in pairs:
                if row < num_added and col < num_grouped and diff_saved[row][col] < self.tag_threshold:
                    poses[col].add(idx, joints[row], tags[row])
                else:
                    pose = Pose(self.num_joints, embd_size)
                    pose.add(idx, joints[row], tags[row])
                    poses.append(pose)

        ans = np.asarray([p.pose for p in poses], dtype=np.float32).reshape(-1, self.num_joints, 2 + 1 + embd_size)
        tags = np.asarray([p.tag for p in poses], dtype=np.float32).reshape(-1, embd_size)
        return ans, tags

    def top_k(self, heatmaps, tags):
        N, K, H, W = heatmaps.shape
        heatmaps = heatmaps.reshape(N, K, -1)
        ind = heatmaps.argpartition(-self.max_num_people, axis=2)[:, :, -self.max_num_people:]
        val_k = np.take_along_axis(heatmaps, ind, axis=2)
        subind = np.argsort(-val_k, axis=2)
        ind = np.take_along_axis(ind, subind, axis=2)
        val_k = np.take_along_axis(val_k, subind, axis=2)

        tags = tags.reshape(N, K, W * H, -1)
        tag_k = [np.take_along_axis(tags[..., i], ind, axis=2) for i in range(tags.shape[3])]
        tag_k = np.stack(tag_k, axis=3)

        y, x = np.divmod(ind, W)
        ind_k = np.stack((x, y), axis=3)

        ans = {'tag_k': tag_k, 'loc_k': ind_k, 'val_k': val_k}
        return ans

    @staticmethod
    def adjust(ans, heatmaps):
        H, W = heatmaps.shape[-2:]
        for n, people in enumerate(ans):
            for person in people:
                for k, joint in enumerate(person):
                    heatmap = heatmaps[n, k]
                    px = int(joint[0])
                    py = int(joint[1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py, px + 1] - heatmap[py, px - 1],
                            heatmap[py + 1, px] - heatmap[py - 1, px]
                        ])
                        joint[:2] += np.sign(diff) * .25
        return ans

    @staticmethod
    def combine(poses, poses_scores, poses_tags, thr=1):
        def is_disjoint(pose_a, pose_b):
            return not np.any(np.logical_and(pose_a[:, 2] > 0, pose_b[:, 2] > 0))

        idx = np.argsort(poses_scores)[::-1]
        poses_scores = poses_scores[idx]
        poses = poses[idx]
        poses_tags = poses_tags[idx]

        n = len(poses)
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            p = poses[i]
            p_tag = poses_tags[i]
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                q = poses[j]
                q_tag = poses_tags[j]
                if not keep[j]:
                    continue
                if is_disjoint(p, q) and np.abs(p_tag - q_tag) < thr:
                    p += q
                    # p = np.where(p[:, 2:3] > q[:, 2:3], p, q)
                    poses[i] = p
                    keep[j] = False
                # if np.abs(poses_tags[i] - poses_tags[j]) < thr:
                #     p = np.where(p[:, 2:3] > q[:, 2:3], p, q)
                #     keep[j] = False
        return poses[keep], poses_scores[keep]

    @staticmethod
    def refine(heatmap, tag, keypoints, pose_tag=None):
        K, H, W = heatmap.shape
        if len(tag.shape) == 3:
            tag = tag[..., None]

        if pose_tag is not None:
            prev_tag = pose_tag
        else:
            tags = []
            for i in range(K):
                if keypoints[i, 2] > 0:
                    x, y = keypoints[i][:2].astype(int)
                    tags.append(tag[i, y, x])
            prev_tag = np.mean(tags, axis=0)

        for i, (_heatmap, _tag) in enumerate(zip(heatmap, tag)):
            if keypoints[i, 2] > 0:
                continue
            # Get position with the closest tag value to the pose tag.
            diff = np.abs(_tag[..., 0] - prev_tag) + 0.5
            diff = diff.astype(np.int32).astype(_heatmap.dtype)
            diff -= _heatmap
            idx = diff.argmin()
            y, x = np.divmod(idx, _heatmap.shape[-1])
            # Corresponding keypoint detection score.
            val = _heatmap[y, x]
            if val > 0:
                keypoints[i, :3] = x, y, 2048
                if 1 < x < W - 1 and 1 < y < H - 1:
                    diff = np.array([
                        _heatmap[y, x + 1] - _heatmap[y, x - 1],
                        _heatmap[y + 1, x] - _heatmap[y - 1, x]
                    ])
                    keypoints[i, :2] += np.sign(diff) * .25

                    # dists = np.linalg(all_keypoints - keypoints[i:i + 1, :2], ord=2, axis=-1, keepdims=False)
                    # print(min(dists))

        return keypoints

    def __call__(self, heatmaps, tags, nms_heatmaps=None):
        ans = self.match(**self.top_k(nms_heatmaps, tags))
        ans, ans_tags = map(list, zip(*ans))

        # for i, people in enumerate(ans):
        #     mask = np.where((people[..., 2] > 0).sum(axis=-1) >= 3)
        #     ans[i] = people[mask]

        if self.do_adjust:
            ans = self.adjust(ans, heatmaps)

        if self.delta != 0.0:
            for people in ans:
                for person in people:
                    for joint in person:
                        joint[:2] += self.delta

        ans = ans[0]
        scores = np.asarray([i[:, 2].mean() for i in ans])
        # scores = []
        # for i in ans:
        #     x = i[:, 2]
        #     scores.append(x[x > 0].mean())
        # scores = np.asarray(scores, dtype=np.float32)

        if self.do_refine:
            heatmap_numpy = heatmaps[0]
            tag_numpy = tags[0]
            for i, pose in enumerate(ans):
                ans[i] = self.refine(heatmap_numpy, tag_numpy, pose, ans_tags[0][i])
        else:
            ans, scores = self.combine(ans, scores, ans_tags[0], 1.0)

        # scores = np.asarray([i[:, 2].mean() for i in ans])

        return ans, scores


class DummyAssociativeEmbeddingDecoder:
    def __init__(self, num_joints, max_num_people, detection_threshold, use_detection_val,
                 ignore_too_much, tag_threshold,
                 adjust=True, refine=True, delta=0.0, joints_order=None):
        self.num_joints = num_joints
        self.max_num_people = max_num_people
        self.detection_threshold = detection_threshold
        self.tag_threshold = tag_threshold
        self.use_detection_val = use_detection_val
        self.ignore_too_much = ignore_too_much

        if self.num_joints == 17 and joints_order is None:
            self.joint_order = (0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16)
        else:
            self.joint_order = list(np.arange(self.num_joints))

        self.do_adjust = adjust
        self.do_refine = refine
        self.delta = delta

    def top_k(self, heatmaps, tags):
        N, K, H, W = heatmaps.shape
        heatmaps = heatmaps.reshape(N, K, -1)
        ind = heatmaps.argpartition(-self.max_num_people, axis=2)[:, :, -self.max_num_people:]
        val_k = np.take_along_axis(heatmaps, ind, axis=2)
        subind = np.argsort(-val_k, axis=2)
        ind = np.take_along_axis(ind, subind, axis=2)
        val_k = np.take_along_axis(val_k, subind, axis=2)

        tags = tags.reshape(N, K, W * H, -1)
        tag_k = [np.take_along_axis(tags[..., i], ind, axis=2) for i in range(tags.shape[3])]
        tag_k = np.stack(tag_k, axis=3)

        y, x = np.divmod(ind, W)
        ind_k = np.stack((x, y), axis=3)

        ans = {'tag_k': tag_k, 'loc_k': ind_k, 'val_k': val_k}
        return ans

    @staticmethod
    def adjust(ans, heatmaps):
        H, W = heatmaps.shape[-2:]
        for n, people in enumerate(ans):
            for person in people:
                for k, joint in enumerate(person):
                    heatmap = heatmaps[n, k]
                    px = int(joint[0])
                    py = int(joint[1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py, px + 1] - heatmap[py, px - 1],
                            heatmap[py + 1, px] - heatmap[py - 1, px]
                        ])
                        joint[:2] += np.sign(diff) * .25
        return ans

    def __call__(self, heatmaps, tags, nms_heatmaps=None):
        joints = self.top_k(nms_heatmaps, tags)
        # print(joints['loc_k'].shape, joints['val_k'].shape)
        poses = np.zeros((self.max_num_people, self.num_joints, 3), dtype=float)
        for k, (joints, scores) in enumerate(zip(joints['loc_k'][0], joints['val_k'][0])):
            for i, (joint, score) in enumerate(zip(joints, scores)):
                poses[i, k, :2] = joint
                poses[i, k, 2] = score
        ans = poses[None]
        # print(ans.shape)

        if self.do_adjust:
            ans = self.adjust(ans, heatmaps)

        if self.delta != 0.0:
            for people in ans:
                for person in people:
                    for joint in person:
                        joint[:2] += self.delta

        ans = ans[0]
        scores = np.asarray([i[:, 2].mean() for i in ans])
        return ans, scores
