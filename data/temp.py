#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
from utils.augmentations import preprocess
from data.config import cfg
import cv2
import math

class WIDERDetection(data.Dataset):
    """docstring for WIDERDetection"""

    def __init__(self, list_file, mode='train'):
        super(WIDERDetection, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in range(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
                label.append(c)
            if len(box) > 0:
                self.fnames.append(line[0])
                self.boxes.append(box)
                self.labels.append(label)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 为了取出所有配对图
        img_path = self.fnames[index]
        # print(f'fname in getitem is {img_path}')
        # exit()
        img = Image.open(img_path)
        # 针对TOG all & Domain adaptation
        # 在数据集中提前加载所有的配对图
        I_light = Image.open(get_paired_path('illuminDay', img_path))
        R_light = Image.open(get_paired_path('perturbDay', img_path))
        I_dark = Image.open(get_paired_path('illuminNight', img_path))
        R_dark = Image.open(get_paired_path('perturbNight', img_path))
        # 将所有图像打包传入新的预处理函数
        im_width, im_height = img.size
        boxes = self.annotransform(
            np.array(self.boxes[index]), im_width, im_height)
        label = np.array(self.labels[index])
        bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()

        (img, I_light, R_light, I_dark, R_dark), targets = sync_preprocess(
            [img, I_light, R_light, I_dark, R_dark], bbox_labels, self.mode, cfg
        )
        
        return img, I_light, R_light, I_dark, R_dark, targets, img_path


    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')

            im_width, im_height = img.size
            boxes = self.annotransform(
                np.array(self.boxes[index]), im_width, im_height)
            label = np.array(self.labels[index])
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()

            # ===== TOG训练时仅缩放，不做任何增强 =====
            # 随机选择插值方式（与原 preprocess 一致）
            interp_mode = [
                Image.BILINEAR, Image.HAMMING, Image.NEAREST,
                Image.BICUBIC, Image.LANCZOS
            ]
            interp_idx = np.random.randint(0, 5)
            img = img.resize((cfg.resize_width, cfg.resize_height),
                            resample=interp_mode[interp_idx])
            # 此时 img 是 PIL Image，尺寸为 (resize_width, resize_height)
            # ========== 新增：将 PIL Image 转为 numpy 并调整维度 ==========
            img = np.array(img)                # 此时 shape: (H, W, C) uint8, RGB 顺序
            img = img.transpose(2, 0, 1)       # 变为 (C, H, W) uint8
            # ============================================================
            # 标注框坐标不变（因为全局缩放，归一化坐标依然有效）
            sample_labels = bbox_labels  # 直接使用
            # ================================

            # 裁剪移到所有图片生成后进行
            # img, sample_labels = preprocess(
            #     img, bbox_labels, self.mode, image_path)

            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break 
            else:
                index = random.randrange(0, self.num_samples)

        
        #img = Image.fromarray(img)
        '''
        draw = ImageDraw.Draw(img)
        w,h = img.size
        for bbox in sample_labels:
            bbox = (bbox[1:] * np.array([w, h, w, h])).tolist()

            draw.rectangle(bbox,outline='red')
        img.save('image.jpg')
        '''
        return torch.from_numpy(img), target, image_path, im_height, im_width

    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    paths = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        paths.append(sample[2])
    return torch.stack(imgs, 0), targets, paths

def get_paired_path(subdir, img_path, base_dir='/home/share/lowdetect/dataset/myWORK/'):
    """从子目录加载图像，返回 [0,1] tensor"""
    fname = os.path.basename(img_path)
    paired_path = os.path.join(base_dir, subdir, fname)
    if not os.path.exists(paired_path):
        raise FileNotFoundError(f"Missing {paired_path}")
    # img = Image.open(full_path).convert('RGB')
    return paired_path
    
class sampler():

    def __init__(self,
                 max_sample,
                 max_trial,
                 min_scale,
                 max_scale,
                 min_aspect_ratio,
                 max_aspect_ratio,
                 min_jaccard_overlap,
                 max_jaccard_overlap,
                 min_object_coverage,
                 max_object_coverage,
                 use_square=False):
        self.max_sample = max_sample
        self.max_trial = max_trial
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_jaccard_overlap = min_jaccard_overlap
        self.max_jaccard_overlap = max_jaccard_overlap
        self.min_object_coverage = min_object_coverage
        self.max_object_coverage = max_object_coverage
        self.use_square = use_square

class bbox():

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

def generate_sample(sampler, image_width, image_height):
    scale = np.random.uniform(sampler.min_scale, sampler.max_scale)
    aspect_ratio = np.random.uniform(sampler.min_aspect_ratio,
                                     sampler.max_aspect_ratio)
    aspect_ratio = max(aspect_ratio, (scale**2.0))
    aspect_ratio = min(aspect_ratio, 1 / (scale**2.0))

    bbox_width = scale * (aspect_ratio**0.5)
    bbox_height = scale / (aspect_ratio**0.5)

    # guarantee a squared image patch after cropping
    if sampler.use_square:
        if image_height < image_width:
            bbox_width = bbox_height * image_height / image_width
        else:
            bbox_height = bbox_width * image_width / image_height

    xmin_bound = 1 - bbox_width
    ymin_bound = 1 - bbox_height
    xmin = np.random.uniform(0, xmin_bound)
    ymin = np.random.uniform(0, ymin_bound)
    xmax = xmin + bbox_width
    ymax = ymin + bbox_height
    sampled_bbox = bbox(xmin, ymin, xmax, ymax)
    return sampled_bbox

def generate_batch_samples(batch_sampler, bbox_labels, image_width,
                           image_height):
    sampled_bbox = []
    for sampler in batch_sampler:
        found = 0
        for i in range(sampler.max_trial):
            if found >= sampler.max_sample:
                break
            sample_bbox = generate_sample(sampler, image_width, image_height)
            if satisfy_sample_constraint(sampler, sample_bbox, bbox_labels):
                sampled_bbox.append(sample_bbox)
                found = found + 1
    return sampled_bbox

def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox.xmin >= object_bbox.xmax or \
            sample_bbox.xmax <= object_bbox.xmin or \
            sample_bbox.ymin >= object_bbox.ymax or \
            sample_bbox.ymax <= object_bbox.ymin:
        return 0
    intersect_xmin = max(sample_bbox.xmin, object_bbox.xmin)
    intersect_ymin = max(sample_bbox.ymin, object_bbox.ymin)
    intersect_xmax = min(sample_bbox.xmax, object_bbox.xmax)
    intersect_ymax = min(sample_bbox.ymax, object_bbox.ymax)
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
        sample_bbox_size + object_bbox_size - intersect_size)
    return overlap

def bbox_coverage(bbox1, bbox2):
    inter_box = intersect_bbox(bbox1, bbox2)
    intersect_size = bbox_area(inter_box)

    if intersect_size > 0:
        bbox1_size = bbox_area(bbox1)
        return intersect_size / bbox1_size
    else:
        return 0.

def satisfy_sample_constraint(sampler, sample_bbox, bbox_labels):
    if sampler.min_jaccard_overlap == 0 and sampler.max_jaccard_overlap == 0:
        has_jaccard_overlap = False
    else:
        has_jaccard_overlap = True
    if sampler.min_object_coverage == 0 and sampler.max_object_coverage == 0:
        has_object_coverage = False
    else:
        has_object_coverage = True

    if not has_jaccard_overlap and not has_object_coverage:
        return True
    found = False
    for i in range(len(bbox_labels)):
        object_bbox = bbox(bbox_labels[i][1], bbox_labels[i][2],
                           bbox_labels[i][3], bbox_labels[i][4])
        if has_jaccard_overlap:
            overlap = jaccard_overlap(sample_bbox, object_bbox)
            if sampler.min_jaccard_overlap != 0 and \
                    overlap < sampler.min_jaccard_overlap:
                continue
            if sampler.max_jaccard_overlap != 0 and \
                    overlap > sampler.max_jaccard_overlap:
                continue
            found = True
        if has_object_coverage:
            object_coverage = bbox_coverage(object_bbox, sample_bbox)
            if sampler.min_object_coverage != 0 and \
                    object_coverage < sampler.min_object_coverage:
                continue
            if sampler.max_object_coverage != 0 and \
                    object_coverage > sampler.max_object_coverage:
                continue
            found = True
        if found:
            return True
    return found

def bbox_area(src_bbox):
    if src_bbox.xmax < src_bbox.xmin or src_bbox.ymax < src_bbox.ymin:
        return 0.
    else:
        width = src_bbox.xmax - src_bbox.xmin
        height = src_bbox.ymax - src_bbox.ymin
        return width * height

def crop_image(img, bbox_labels, sample_bbox, image_width, image_height,
               resize_width, resize_height, min_face_size):
    sample_bbox = clip_bbox(sample_bbox)
    xmin = int(sample_bbox.xmin * image_width)
    xmax = int(sample_bbox.xmax * image_width)
    ymin = int(sample_bbox.ymin * image_height)
    ymax = int(sample_bbox.ymax * image_height)

    sample_img = img[ymin:ymax, xmin:xmax]
    resize_val = resize_width
    sample_labels = transform_labels_sampling(bbox_labels, sample_bbox,
                                              resize_val, min_face_size)
    return sample_img, sample_labels

def intersect_bbox(bbox1, bbox2):
    if bbox2.xmin > bbox1.xmax or bbox2.xmax < bbox1.xmin or \
            bbox2.ymin > bbox1.ymax or bbox2.ymax < bbox1.ymin:
        intersection_box = bbox(0.0, 0.0, 0.0, 0.0)
    else:
        intersection_box = bbox(
            max(bbox1.xmin, bbox2.xmin),
            max(bbox1.ymin, bbox2.ymin),
            min(bbox1.xmax, bbox2.xmax), min(bbox1.ymax, bbox2.ymax))
    return intersection_box

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def anchor_crop_image_sampling(img, bbox_labels, scale_array, img_width, img_height):
    mean = np.array([104, 117, 123], dtype=np.float32)
    maxSize = 12000  # max size
    infDistance = 9999999
    bbox_labels = np.array(bbox_labels)
    scale = np.array([img_width, img_height, img_width, img_height])

    boxes = bbox_labels[:, 1:5] * scale
    labels = bbox_labels[:, 0]

    boxArea = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    rand_idx = np.random.randint(len(boxArea))
    rand_Side = boxArea[rand_idx] ** 0.5

    distance = infDistance
    anchor_idx = 5
    for i, anchor in enumerate(scale_array):
        if abs(anchor - rand_Side) < distance:
            distance = abs(anchor - rand_Side)
            anchor_idx = i

    target_anchor = random.choice(scale_array[0:min(anchor_idx + 1, 5) + 1])
    ratio = float(target_anchor) / rand_Side
    ratio = ratio * (2**random.uniform(-1, 1))

    if int(img_height * ratio * img_width * ratio) > maxSize * maxSize:
        ratio = (maxSize * maxSize / (img_height * img_width))**0.5

    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC,
                      cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = random.choice(interp_methods)
    image = cv2.resize(img, None, None, fx=ratio,
                       fy=ratio, interpolation=interp_method)

    boxes[:, 0] *= ratio
    boxes[:, 1] *= ratio
    boxes[:, 2] *= ratio
    boxes[:, 3] *= ratio

    height, width, _ = image.shape

    # 【新增】：在这里记录随机缩放参数
    transform_params = {
        'ratio': ratio,
        'interp_method': interp_method,
        'is_cropped': False,
        'choice_box': None
    }
    sample_boxes = []
    xmin = boxes[rand_idx, 0]
    ymin = boxes[rand_idx, 1]
    bw = (boxes[rand_idx, 2] - boxes[rand_idx, 0] + 1)
    bh = (boxes[rand_idx, 3] - boxes[rand_idx, 1] + 1)
    w = h = 640

    for _ in range(50):
        if w < max(height, width):
            if bw <= w:
                w_off = random.uniform(xmin + bw - w, xmin)
            else:
                w_off = random.uniform(xmin, xmin + bw - w)

            if bh <= h:
                h_off = random.uniform(ymin + bh - h, ymin)
            else:
                h_off = random.uniform(ymin, ymin + bh - h)
        else:
            w_off = random.uniform(width - w, 0)
            h_off = random.uniform(height - h, 0)

        w_off = math.floor(w_off)
        h_off = math.floor(h_off)

        # convert to integer rect x1,y1,x2,y2
        rect = np.array(
            [int(w_off), int(h_off), int(w_off + w), int(h_off + h)])
        # keep overlap with gt box IF center in sampled patch
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        # mask in all gt boxes that above and to the left of centers
        m1 = (rect[0] <= boxes[:, 0]) * (rect[1] <= boxes[:, 1])
        # mask in all gt boxes that under and to the right of centers
        m2 = (rect[2] >= boxes[:, 2]) * (rect[3] >= boxes[:, 3])
        # mask in that both m1 and m2 are true
        mask = m1 * m2
        overlap = jaccard_numpy(boxes, rect)
        # have any valid boxes? try again if not
        if not mask.any() and not overlap.max() > 0.7:
            continue
        else:
            sample_boxes.append(rect)

    sampled_labels = []

    if len(sample_boxes) > 0:
        choice_idx = np.random.randint(len(sample_boxes))
        choice_box = sample_boxes[choice_idx]
        # print('crop the box :',choice_box)

        # 【新增】：记录决定使用的裁剪框
        transform_params['is_cropped'] = True
        transform_params['choice_box'] = choice_box

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        m1 = (choice_box[0] < centers[:, 0]) * \
            (choice_box[1] < centers[:, 1])
        m2 = (choice_box[2] > centers[:, 0]) * \
            (choice_box[3] > centers[:, 1])
        mask = m1 * m2
        current_boxes = boxes[mask, :].copy()
        current_labels = labels[mask]
        current_boxes[:, :2] -= choice_box[:2]
        current_boxes[:, 2:] -= choice_box[:2]

        if choice_box[0] < 0 or choice_box[1] < 0:
            new_img_width = width if choice_box[0] >= 0 else width - choice_box[0]
            new_img_height = height if choice_box[1] >= 0 else height - choice_box[1]
            image_pad = np.zeros( (new_img_height, new_img_width, 3), dtype=float)
            image_pad[:, :, :] = mean
            start_left = 0 if choice_box[0] >= 0 else -choice_box[0]
            start_top = 0 if choice_box[1] >= 0 else -choice_box[1]
            image_pad[start_top:, start_left:, :] = image

            choice_box_w = choice_box[2] - choice_box[0]
            choice_box_h = choice_box[3] - choice_box[1]

            start_left = choice_box[0] if choice_box[0] >= 0 else 0
            start_top = choice_box[1] if choice_box[1] >= 0 else 0

            end_right = start_left + choice_box_w
            end_bottom = start_top + choice_box_h

            current_image = image_pad[start_top:end_bottom, start_left:end_right, :].copy()
            image_height, image_width, _ = current_image.shape

            if cfg.filter_min_face:
                bbox_w = current_boxes[:, 2] - current_boxes[:, 0]
                bbox_h = current_boxes[:, 3] - current_boxes[:, 1]
                bbox_area = bbox_w * bbox_h
                mask = bbox_area > (cfg.min_face_size * cfg.min_face_size)
                current_boxes = current_boxes[mask]
                current_labels = current_labels[mask]
                for i in range(len(current_boxes)):
                    sample_label = []
                    sample_label.append(current_labels[i])
                    sample_label.append(current_boxes[i][0] / image_width)
                    sample_label.append(current_boxes[i][1] / image_height)
                    sample_label.append(current_boxes[i][2] / image_width)
                    sample_label.append(current_boxes[i][3] / image_height)
                    sampled_labels += [sample_label]
                sampled_labels = np.array(sampled_labels)
            else:
                current_boxes /= np.array([image_width,image_height, image_width, image_height])
                sampled_labels = np.hstack((current_labels[:, np.newaxis], current_boxes))

            # 【修改】：增加了返回 transform_params
            return current_image, sampled_labels, transform_params

        current_image = image[choice_box[1]:choice_box[3], choice_box[0]:choice_box[2], :].copy()
        image_height, image_width, _ = current_image.shape

        if cfg.filter_min_face:
            bbox_w = current_boxes[:, 2] - current_boxes[:, 0]
            bbox_h = current_boxes[:, 3] - current_boxes[:, 1]
            bbox_area = bbox_w * bbox_h
            mask = bbox_area > (cfg.min_face_size * cfg.min_face_size)
            current_boxes = current_boxes[mask]
            current_labels = current_labels[mask]
            for i in range(len(current_boxes)):
                sample_label = []
                sample_label.append(current_labels[i])
                sample_label.append(current_boxes[i][0] / image_width)
                sample_label.append(current_boxes[i][1] / image_height)
                sample_label.append(current_boxes[i][2] / image_width)
                sample_label.append(current_boxes[i][3] / image_height)
                sampled_labels += [sample_label]
            sampled_labels = np.array(sampled_labels)
        else:
            current_boxes /= np.array([image_width, image_height, image_width, image_height])
            sampled_labels = np.hstack( (current_labels[:, np.newaxis], current_boxes))

        # 【修改】：增加了返回 transform_params
        return current_image, sampled_labels, transform_params
    else:
        image_height, image_width, _ = image.shape
        if cfg.filter_min_face:
            bbox_w = boxes[:, 2] - boxes[:, 0]
            bbox_h = boxes[:, 3] - boxes[:, 1]
            bbox_area = bbox_w * bbox_h
            mask = bbox_area > (cfg.min_face_size * cfg.min_face_size)
            boxes = boxes[mask]
            labels = labels[mask]
            for i in range(len(boxes)):
                sample_label = []
                sample_label.append(labels[i])
                sample_label.append(boxes[i][0] / image_width)
                sample_label.append(boxes[i][1] / image_height)
                sample_label.append(boxes[i][2] / image_width)
                sample_label.append(boxes[i][3] / image_height)
                sampled_labels += [sample_label]
            sampled_labels = np.array(sampled_labels)
        else:
            boxes /= np.array([image_width, image_height, image_width, image_height])
            sampled_labels = np.hstack( (labels[:, np.newaxis], boxes))

        # 【修改】：增加了返回 transform_params
        return image, sampled_labels, transform_params

def apply_anchor_transform_to_paired(img_array, transform_params):
    """
    接收 numpy array 格式的配对图像，并按照主图的随机参数执行相同的变换。
    """
    ratio = transform_params['ratio']
    interp_method = transform_params['interp_method']
    
    # 1. 相同的等比例缩放
    image = cv2.resize(img_array, None, None, fx=ratio, fy=ratio, interpolation=interp_method)
    
    # 2. 如果主图没裁剪，直接返回缩放后的图
    if not transform_params['is_cropped']:
        return image
        
    # 3. 如果主图执行了裁剪
    choice_box = transform_params['choice_box']
    if choice_box[0] < 0 or choice_box[1] < 0:
        # 情况 A: 越界，需要 Padding
        height, width, _ = image.shape
        new_img_width = width if choice_box[0] >= 0 else width - choice_box[0]
        new_img_height = height if choice_box[1] >= 0 else height - choice_box[1]
        
        # 注意：光照/反射图填充黑色 (0) 是最安全的，不破坏物理特性
        image_pad = np.zeros((new_img_height, new_img_width, 3), dtype=image.dtype)
        
        start_left = 0 if choice_box[0] >= 0 else -choice_box[0]
        start_top = 0 if choice_box[1] >= 0 else -choice_box[1]
        image_pad[start_top:, start_left:, :] = image

        choice_box_w = choice_box[2] - choice_box[0]
        choice_box_h = choice_box[3] - choice_box[1]

        start_left = choice_box[0] if choice_box[0] >= 0 else 0
        start_top = choice_box[1] if choice_box[1] >= 0 else 0
        end_right = start_left + choice_box_w
        end_bottom = start_top + choice_box_h
        
        current_image = image_pad[start_top:end_bottom, start_left:end_right, :].copy()
        return current_image
    else:
        # 情况 B: 正常裁剪
        current_image = image[choice_box[1]:choice_box[3], choice_box[0]:choice_box[2], :].copy()
        return current_image
    
def sync_preprocess(image_list, bbox_labels, mode, cfg):
    """
    image_list: [img_main, I_light, R_light, I_dark, R_dark] (全为 PIL Image)
    """
    img_main = image_list[0]
    paired_imgs = image_list[1:]
    img_width, img_height = img_main.size
    sampled_labels = bbox_labels

    if mode == 'train':
        # ====== 1. 颜色失真 (Distort) ======
        # 【关键】：Distort 只能作用于原图 (img_main)。
        # 绝对不能作用于 I_light 和 R_light，否则会破坏 Retinex 物理守恒定律！
        # if cfg.apply_distort:
        #     img_main = distort_image(img_main)
            
        # 转换为 numpy 以便进行几何变换
        img_main = np.array(img_main)
        paired_imgs = [np.array(img) for img in paired_imgs]

        # ====== 2. 图像扩展 (Expand) ======
        # if cfg.apply_expand:
        #     # 仅对主图计算 expand 随机参数
        #     # img_main, bbox_labels, img_width, img_height, left, top = custom_expand_return_params( img_main, bbox_labels, img_width, img_height) # 你需要修改底层 expand 函数返回 left 和 top
        #     print(f'there is not suitable forw expand')
            
        #     # 使用算出的 left, top 同步 expand 其他配对图
        #     new_paired = []
        #     for p_img in paired_imgs:
        #         # 光照图/反射图的填充背景建议用 0 (黑色) 而不是均值
        #         exp_p = np.zeros((img_height, img_width, 3), dtype=p_img.dtype)
        #         exp_p[top:top+p_img.shape[0], left:left+p_img.shape[1]] = p_img
        #         new_paired.append(exp_p)
        #     paired_imgs = new_paired

        # ====== 3. 同步裁剪 (Anchor Crop) ======
        batch_sampler = []
        prob = np.random.uniform(0., 1.)
        if prob > cfg.data_anchor_sampling_prob and cfg.anchor_sampling:
            scale_array = np.array([16, 32, 64, 128, 256, 512])

            # 使用上一次我们修改后的 anchor_crop 返回 transform_params
            img_main, sampled_labels, transform_params = anchor_crop_image_sampling(
                img_main, sampled_labels, scale_array, img_width, img_height)
            
            # 同步应用到其余配对图
            paired_imgs = [apply_anchor_transform_to_paired(p, transform_params) for p in paired_imgs]
            
        else:
            # 常规 SSD 裁剪：本身就生成了 sampled_bbox，直接复用！
            # 标准 SSD Crop
            batch_sampler = []
            batch_sampler.append(sampler(1, 50, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, True))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, True))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, True))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, True))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, True))
            
            sampled_bbox = generate_batch_samples(batch_sampler, bbox_labels, img_width, img_height)
            if len(sampled_bbox) > 0:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                box = sampled_bbox[idx]
                
                # 同步对所有图片使用同一个 box 进行裁剪
                img_main, sampled_labels = crop_image(img_main, bbox_labels, box, img_width, img_height, cfg.resize_width, cfg.resize_height, cfg.min_face_size)
                
                new_paired = []
                for p_img in paired_imgs:
                    # 使用同样的 box 裁剪其余图像 (忽略返回的 label)
                    p_cropped, _ = crop_image(p_img, bbox_labels.copy(), box, img_width, img_height, cfg.resize_width, cfg.resize_height, cfg.min_face_size)
                    new_paired.append(p_cropped)
                paired_imgs = new_paired

        # 转回 PIL 供后续 Resize
        img_main = Image.fromarray(img_main.astype('uint8'))
        paired_imgs = [Image.fromarray(p.astype('uint8')) for p in paired_imgs]

    # ====== 4. 同步 Resize ======
    interp_mode = [Image.BILINEAR, Image.HAMMING, Image.NEAREST, Image.BICUBIC, Image.LANCZOS]
    interp_indx = np.random.randint(0, 5)
    
    img_main = img_main.resize((cfg.resize_width, cfg.resize_height), resample=interp_mode[interp_indx])
    paired_imgs = [p.resize((cfg.resize_width, cfg.resize_height), resample=interp_mode[interp_indx]) for p in paired_imgs]

    img_main = np.array(img_main)
    paired_imgs = [np.array(p) for p in paired_imgs]

    # ====== 5. 同步水平翻转 (Mirror) ======
    if mode == 'train':
        if int(np.random.uniform(0, 2)) == 1:
            img_main = img_main[:, ::-1, :]
            paired_imgs = [p[:, ::-1, :] for p in paired_imgs]
            for i in range(len(sampled_labels)):
                tmp = sampled_labels[i][1]
                sampled_labels[i][1] = 1 - sampled_labels[i][3]
                sampled_labels[i][3] = 1 - tmp

    # ====== 6. 格式化输出 ======
    def format_tensor(img, apply_mean=False):
        img = to_chw_bgr(img).astype('float32')
        if apply_mean:
            img -= cfg.img_mean # 检测器主图需要减均值
        img = img[[2, 1, 0], :, :] # 转 RGB
        return img

    img_main = format_tensor(img_main, apply_mean=True)
    # 对于增强网络的图，不减均值，保持物理意义，转为 0~1 的张量
    paired_imgs = [format_tensor(p, apply_mean=False) / 255.0 for p in paired_imgs] 

    return [img_main] + paired_imgs, sampled_labels

def clip_bbox(src_bbox):
    src_bbox.xmin = max(min(src_bbox.xmin, 1.0), 0.0)
    src_bbox.ymin = max(min(src_bbox.ymin, 1.0), 0.0)
    src_bbox.xmax = max(min(src_bbox.xmax, 1.0), 0.0)
    src_bbox.ymax = max(min(src_bbox.ymax, 1.0), 0.0)
    return src_bbox

def transform_labels_sampling(bbox_labels, sample_bbox, resize_val,
                              min_face_size):
    sample_labels = []
    for i in range(len(bbox_labels)):
        sample_label = []
        object_bbox = bbox(bbox_labels[i][1], bbox_labels[i][2],
                           bbox_labels[i][3], bbox_labels[i][4])
        if not meet_emit_constraint(object_bbox, sample_bbox):
            continue
        proj_bbox = project_bbox(object_bbox, sample_bbox)
        if proj_bbox:
            real_width = float((proj_bbox.xmax - proj_bbox.xmin) * resize_val)
            real_height = float((proj_bbox.ymax - proj_bbox.ymin) * resize_val)
            if real_width * real_height < float(min_face_size * min_face_size):
                continue
            else:
                sample_label.append(bbox_labels[i][0])
                sample_label.append(float(proj_bbox.xmin))
                sample_label.append(float(proj_bbox.ymin))
                sample_label.append(float(proj_bbox.xmax))
                sample_label.append(float(proj_bbox.ymax))
                sample_label = sample_label + bbox_labels[i][5:]
                sample_labels.append(sample_label)

    return sample_labels

def to_chw_bgr(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.
    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image

def meet_emit_constraint(src_bbox, sample_bbox):
    center_x = (src_bbox.xmax + src_bbox.xmin) / 2
    center_y = (src_bbox.ymax + src_bbox.ymin) / 2
    if center_x >= sample_bbox.xmin and \
            center_x <= sample_bbox.xmax and \
            center_y >= sample_bbox.ymin and \
            center_y <= sample_bbox.ymax:
        return True
    return False

def project_bbox(object_bbox, sample_bbox):
    if object_bbox.xmin >= sample_bbox.xmax or \
       object_bbox.xmax <= sample_bbox.xmin or \
       object_bbox.ymin >= sample_bbox.ymax or \
       object_bbox.ymax <= sample_bbox.ymin:
        return False
    else:
        proj_bbox = bbox(0, 0, 0, 0)
        sample_width = sample_bbox.xmax - sample_bbox.xmin
        sample_height = sample_bbox.ymax - sample_bbox.ymin
        proj_bbox.xmin = (object_bbox.xmin - sample_bbox.xmin) / sample_width
        proj_bbox.ymin = (object_bbox.ymin - sample_bbox.ymin) / sample_height
        proj_bbox.xmax = (object_bbox.xmax - sample_bbox.xmin) / sample_width
        proj_bbox.ymax = (object_bbox.ymax - sample_bbox.ymin) / sample_height
        proj_bbox = clip_bbox(proj_bbox)
        if bbox_area(proj_bbox) > 0:
            return proj_bbox
        else:
            return False



if __name__ == '__main__':
    from config import cfg
    dataset = WIDERDetection(cfg.FACE.TRAIN_FILE)
    #for i in range(len(dataset)):
    dataset.pull_item(14)
