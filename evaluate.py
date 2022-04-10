import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pylab as pl

from tensorpack import PredictConfig
from tensorpack import get_model_loader
from tensorpack import SimpleDatasetPredictor
from tensorpack.utils import viz
from tensorpack.utils.fs import mkdir_p

from data_loader import get_data
from os.path import join as ospj
import dataflow
from tensorpack.utils import logger
from colors import COLORS


def vis(index, convmaps, name, option):
    # b, c, h, w = np.shape(convmaps)
    convmap = convmaps[index, :, :, :]
    resizemap = cv2.resize(convmap, (option.final_size, option.final_size))
    heatmap = viz.intensity_to_rgb(resizemap, cmap='gray', normalize=True)
    fname = 'train_log/{}/pool1.jpg'.format(option.log_dir)
    cv2.imwrite(fname, heatmap)


def get_model(model, ckpt_name, option):
    model_path = ospj('train_log', option.log_dir, ckpt_name)
    ds = get_data('val', option)
    # ds = get_data('train', option)
    if option.method_name in ['AAE', 'SAE', 'AAE_SAE']:
        pred_config = PredictConfig(model=model,
                                    session_init=get_model_loader(model_path),
                                    input_names=['input', 'label', 'bbox'],
                                    output_names=[
                                        'wrong-top1', 'top5', 'actmap', 'grad',
                                        'top5_1x3', 'actmap_1x3', 'grad_1x3',
                                        'top5_3x1', 'actmap_3x1', 'grad_3x1'
                                    ],
                                    return_input=True)
    else:
        pred_config = PredictConfig(
            model=model,
            session_init=get_model_loader(model_path),
            input_names=['input', 'label', 'bbox'],
            output_names=['wrong-top1', 'top5', 'actmap', 'grad'],
            return_input=True)

    return SimpleDatasetPredictor(pred_config, ds)


def get_meta(option):
    if option.dataset_name == 'ILSVRC':
        meta = dataflow.ImagenetMeta().get_synset_words_1000()
        meta_labels = dataflow.ImagenetMeta().get_synset_1000()
    elif option.dataset_name == 'CUB':
        meta = dataflow.CUBMeta().get_synset_words_1000()
        meta_labels = dataflow.CUBMeta().get_synset_1000()
    else:
        raise KeyError("Unavailable dataset: {}".format(option.dataset_name))

    return meta, meta_labels


def get_log_dir(option):
    threshold_idx = int(option.cam_threshold * 100)
    dirname = ospj('train_log', option.log_dir, 'result', str(threshold_idx))
    if not os.path.isdir(dirname):
        mkdir_p(dirname)
    return dirname, threshold_idx


def get_cam(index, averaged_gradients, convmaps, option):
    batch_size, channel_size, height, width = np.shape(convmaps)

    averaged_gradient = averaged_gradients[index]
    convmap = convmaps[index, :, :, :]
    mergedmap = np.matmul(averaged_gradient,
                          convmap.reshape((channel_size, -1))). \
        reshape(height, width)
    mergedmap = cv2.resize(mergedmap, (option.final_size, option.final_size))
    heatmap = viz.intensity_to_rgb(mergedmap, normalize=True)
    heatmap_gist_ncar = viz.intensity_to_rgb(mergedmap,
                                             cmap='hsv',
                                             normalize=True)
    return heatmap, heatmap_gist_ncar


def get_cam_res(index, averaged_gradients, convmaps, option):
    batch_size, channel_size, height, width = np.shape(convmaps[0])

    for _i, con in enumerate(convmaps):
        averaged_gradient = averaged_gradients[_i][index]
        convmap = con[index, :, :, :]
        if _i == 0:
            mergedmap = np.matmul(averaged_gradient,
                                  convmap.reshape(
                                      (channel_size,
                                       -1))).reshape(height, width)
        else:
            mergedmap_temp = np.matmul(averaged_gradient,
                                       convmap.reshape(
                                           (channel_size,
                                            -1))).reshape(height, width)
            mergedmap = np.maximum(mergedmap, mergedmap_temp)

    mergedmap = cv2.resize(mergedmap, (option.final_size, option.final_size))
    heatmap = viz.intensity_to_rgb(mergedmap, normalize=True)
    heatmap_gist_ncar = viz.intensity_to_rgb(mergedmap,
                                             cmap='hsv',
                                             normalize=True)
    return heatmap, heatmap_gist_ncar


def temp_generate_heatmap(index, averaged_gradients, convmaps, option):
    batch_size, channel_size, height, width = np.shape(convmaps)

    averaged_gradient = averaged_gradients[index]
    convmap = convmaps[index, :, :, :]
    mergedmap = np.matmul(averaged_gradient,
                          convmap.reshape((channel_size, -1))). \
        reshape(height, width)
    mergedmap = cv2.resize(mergedmap, (option.final_size, option.final_size))
    heatmap_gist_ncar = viz.intensity_to_rgb(mergedmap,
                                             cmap='hsv',
                                             normalize=True)
    return mergedmap, heatmap_gist_ncar


def save_3_branch_features(index, images, convs, grads, g, c, option,
                           cls_name):
    image = images[index].astype('uint8')

    merge_h, h = temp_generate_heatmap(index, grads[0], convs[0], option)
    merge_1x3, h_1x3 = temp_generate_heatmap(index, grads[1], convs[1], option)
    merge_3x1, h3x1 = temp_generate_heatmap(index, grads[2], convs[2], option)
    # merge = np.maximum(merge_h, merge_1x3)
    # merge = np.maximum(merge, merge_3x1)
    # merge = viz.intensity_to_rgb(merge, cmap='gist_ncar', normalize=True)
    _, merge = temp_generate_heatmap(index, g, c, option)
    heats = []
    for i in [h, h_1x3, h3x1, merge]:
        heat = images[index] * 0.5 + i * 0.5
        heat = heat.astype('uint8')
        heats.append(heat)

    concat = np.concatenate((image, heats[0], heats[1], heats[2], heats[3]),
                            axis=1)

    p = 'train_log/{}/result/branch_map'.format(option.log_dir)
    if not os.path.exists(p):
        os.makedirs(p)
    fname = os.path.join(p, '{}.jpg'.format(cls_name))
    cv2.imwrite(fname, concat)
    # for h, t in zip([image, mergeh2, h2, h2_1x3, h2_3x1],
    #                 ['image', 'merge', '3x3', '1x3', '3x1']):
    #     p = 'train_log/{}/result/branch_map'.format(option.log_dir)
    #     if not os.path.exists(p):
    #         os.makedirs(p)
    #     # concat = np.concatenate((image, heatmap, blend), axis=1)
    #     fname = os.path.join(p, '{}_{}.jpg'.format(cls_name, t))
    #     cv2.imwrite(fname, h)


def get_estimated_box(heatmap, option):
    gray_heatmap = cv2.cvtColor(heatmap.astype('uint8'), cv2.COLOR_RGB2GRAY)
    threshold_value = int(np.max(gray_heatmap) * option.cam_threshold)

    _, thresholded_gray_heatmap = cv2.threshold(gray_heatmap, threshold_value,
                                                255, cv2.THRESH_TOZERO)
    _, contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [0, 0, 1, 1]

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    estimated_box = [x, y, x + w, y + h]

    return estimated_box


def get_estimated_box_nms_merge(heatmap, option):
    estimated_box_top = get_estimated_box_from_th(heatmap,
                                                  option.cam_threshold)
    # boxes = []
    new_box = estimated_box_top
    for i in list(np.arange(0, 0.1, 0.01)):
        bbox = get_estimated_box_from_th(heatmap, i)
        # boxes.append(bbox)
        #     continue
        # draw_boxes(index, images, boxes)
        iou = nms(estimated_box_top, bbox)
        if iou >= 0.7:
            new_box = np.minimum(bbox, estimated_box_top)
    estimated_box_top = merge_box(new_box, estimated_box_top)
    return estimated_box_top


def merge_box(b1, b2):
    x1 = (b1[0] + b2[0]) // 2
    y1 = (b1[1] + b2[1]) // 2
    x2 = (b1[2] + b2[2]) // 2
    y2 = (b1[3] + b2[3]) // 2
    return [x1, y1, x2, y2]


def get_estimated_box_draw_all(index, images, heatmap, option, cls_name):
    estimated_box_top = get_estimated_box_from_th(heatmap,
                                                  option.cam_threshold)
    boxes = []
    for i in list(np.arange(0, 1, 0.001)):
        bbox = get_estimated_box_from_th(heatmap, i)
        boxes.append(bbox)
    l = len(boxes)
    boxes = np.unique(boxes, axis=0)
    image_with_bbox = draw_boxes(index, images, boxes)
    # cv2.rectangle(image_with_bbox,
    #               (estimated_box_top[0], estimated_box_top[1]),
    #               (estimated_box_top[2], estimated_box_top[3]), (0, 255, 0), 2)
    # n = cls_name + str(len(boxes)) + '_.jpg'
    n = "{}_{}vs{}.jpg".format(cls_name, str(l), str(len(boxes)))
    cv2.imwrite(
        os.path.join('./out_with_box/', option.dataset_name, option.arch_name,
                     n), image_with_bbox)
    # return len(boxes)


def draw_boxes(index, images, bbox):
    image_with_bbox = images[index].astype('uint8')
    for _i, b in enumerate(bbox):
        cv2.rectangle(image_with_bbox, (b[0], b[1]), (b[2], b[3]), COLORS[_i],
                      1)
    return image_with_bbox
    # cv2.imwrite('./out_with_box/' + str(index) + '.jpg', image_with_bbox)


def nms(b1, b2):
    areas_b1 = (b1[3] - b1[1] + 1) * (b1[2] - b1[0] + 1)
    areas_b2 = (b2[3] - b2[1] + 1) * (b2[2] - b2[0] + 1)
    x11 = np.maximum(b1[0], b2[0])
    y11 = np.maximum(b1[1], b2[1])
    x22 = np.minimum(b1[2], b2[2])
    y22 = np.minimum(b1[3], b2[3])
    w = np.maximum(0.0, x22 - x11 + 1)
    h = np.maximum(0.0, y22 - y11 + 1)
    overlaps = w * h
    iou = overlaps / (areas_b1 + areas_b2 - overlaps)
    return iou


def get_estimated_box_from_th(heatmap, th):
    gray_heatmap = cv2.cvtColor(heatmap.astype('uint8'), cv2.COLOR_RGB2GRAY)
    threshold_value = int(np.max(gray_heatmap) * th)

    _, thresholded_gray_heatmap = cv2.threshold(gray_heatmap, threshold_value,
                                                255, cv2.THRESH_TOZERO)
    _, contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [0, 0, 1, 1]

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    estimated_box = [x, y, x + w, y + h]

    return estimated_box


def get_gt_box(index, bbox):
    gt_x_a = int(bbox[index][0][0])
    gt_y_a = int(bbox[index][0][1])
    gt_x_b = int(bbox[index][1][0])
    gt_y_b = int(bbox[index][1][1])

    gt_box = [gt_x_a, gt_y_a, gt_x_b, gt_y_b]
    return gt_box


def draw_images_with_boxes(index, images, heatmap, estimated_box, gt_box):
    image_with_bbox = images[index].astype('uint8')
    cv2.rectangle(image_with_bbox, (estimated_box[0], estimated_box[1]),
                  (estimated_box[2], estimated_box[3]), (0, 255, 0), 2)
    cv2.rectangle(image_with_bbox, (gt_box[0], gt_box[1]),
                  (gt_box[2], gt_box[3]), (0, 0, 255), 2)
    blend = images[index] * 0.5 + heatmap * 0.5
    blend = blend.astype('uint8')
    heatmap = heatmap.astype('uint8')
    concat = np.concatenate((image_with_bbox, heatmap, blend), axis=1)
    return image_with_bbox, concat


def compute_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = np.maximum(0,
                            (x_b - x_a + 1)) * np.maximum(0, (y_b - y_a + 1))
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou


class LocEvaluator(object):
    def __init__(self):
        self.cnt = 0
        self.cnt_false = 0
        self.hit_known = 0
        self.hit_top1 = 0

        self.top1_cls = None
        self.gt_known_loc = None
        self.top1_loc = None

    def accumulate_acc(self, estimated_box, gt_box, wrongs, index):
        iou = compute_iou(estimated_box, gt_box)
        if wrongs[index]:
            self.cnt_false += 1
        if iou > 0.5 or iou == 0.5:
            self.hit_known += 1
        if (iou > 0.5 or iou == 0.5) and not wrongs[index]:
            self.hit_top1 += 1
        self.cnt += 1

    def compute_acc(self):
        self.top1_cls = 1 - self.cnt_false / self.cnt
        self.gt_known_loc = self.hit_known / self.cnt
        self.top1_loc = self.hit_top1 / self.cnt

    def save_img(self, threshold_idx, classname, concat, option):
        fname = 'train_log/{}/result/{}/cam{}-{}.jpg'.format(
            option.log_dir, threshold_idx, self.cnt, classname)
        cv2.imwrite(fname, concat)

    def print_acc(self, threshold_idx, option):
        fname = 'train_log/{}/result/{}/Loc.txt'. \
            format(option.log_dir, threshold_idx)
        with open(fname, 'w') as f:
            line = 'cls: {}\ngt_loc: {}\ntop1_loc: {}'. \
                format(self.top1_cls, self.gt_known_loc, self.top1_loc)
            f.write(line)
        print('thr: {}\n'.format(float(threshold_idx) / 100) + line)
        logger.info('thr: {}\n{}\n{}\n{}\n'.format(
            float(threshold_idx) / 100, self.top1_cls, self.gt_known_loc,
            self.top1_loc))


def evaluate(model, ckpt_name, option):
    pred = get_model(model, ckpt_name, option)
    meta, meta_labels = get_meta(option)
    dirname, threshold_idx = get_log_dir(option)

    evaluator = LocEvaluator()

    for inputs, outputs in pred.get_result():
        images, labels, bbox = inputs
        wrongs, top5, convmaps, gradients = outputs

        if option.is_data_format_nhwc:
            convmaps = np.transpose(convmaps, [0, 3, 1, 2])
            gradients = np.transpose(gradients, [0, 3, 1, 2])

        averaged_gradients = np.mean(gradients, axis=(2, 3))

        for i in range(np.shape(convmaps)[0]):
            heatmap, heatmap_gist_ncar = get_cam(i, averaged_gradients,
                                                 convmaps, option)
            estimated_box = get_estimated_box(heatmap, option)
            gt_box = get_gt_box(i, bbox)
            evaluator.accumulate_acc(estimated_box, gt_box, wrongs, i)
            bbox_img, concat = draw_images_with_boxes(i, images, heatmap,
                                                      estimated_box, gt_box)
            bbox_img, concat_gist_ncar = draw_images_with_boxes(
                i, images, heatmap_gist_ncar, estimated_box, gt_box)
            cls_name = meta[meta_labels[labels[i]]].split(',')[0]

            if evaluator.cnt < 500:
                evaluator.save_img(threshold_idx, cls_name + "_color",
                                   concat_gist_ncar, option)
                # evaluator.save_img(threshold_idx, cls_name, concat, option)

            if evaluator.cnt == option.number_of_val:
                evaluator.compute_acc()
                evaluator.print_acc(threshold_idx, option)
                return


def evaluate_vgg16(model, ckpt_name, option):
    pred = get_model(model, ckpt_name, option)
    meta, meta_labels = get_meta(option)
    dirname, threshold_idx = get_log_dir(option)

    evaluator = LocEvaluator()
    for inputs, outputs in pred.get_result():
        images, labels, bbox = inputs
        wrongs, top5, convmaps, gradients, top5_1x3, actmap_1x3, grad_1x3, top5_3x1, actmap_3x1, grad_3x1 = outputs
        conv = [convmaps, actmap_1x3, actmap_3x1]
        grads = [gradients, grad_1x3, grad_3x1]

        if option.is_data_format_nhwc:
            for _id, c in enumerate(conv):
                c = np.transpose(c, [0, 3, 1, 2])
                grads[_id] = np.transpose(grads[_id], [0, 3, 1, 2])
        averaged_gradients = np.mean(grads[0], axis=(2, 3))
        averaged_gradients_3x3 = np.mean(grads[0], axis=(2, 3))
        averaged_gradients_1x3 = np.mean(grads[1], axis=(2, 3))
        averaged_gradients_3x1 = np.mean(grads[2], axis=(2, 3))
        if False:
            averaged_gradients = np.maximum(averaged_gradients,
                                            averaged_gradients_1x3)
            averaged_gradients = np.maximum(averaged_gradients,
                                            averaged_gradients_3x1)
        else:
            averaged_gradients = (averaged_gradients + averaged_gradients_1x3 +
                                  averaged_gradients_3x1) / 3
            convmaps = (conv[0] + conv[1] + conv[2]) / 3
        convmaps = np.maximum(conv[0], conv[1])
        convmaps = np.maximum(convmaps, conv[2])
        for i in range(np.shape(conv[0])[0]):
            if option.arch_name == 'resnet50_se':
                heatmap, heatmap_gist_ncar = get_cam_res(
                    i, [
                        averaged_gradients, averaged_gradients_1x3,
                        averaged_gradients_3x1
                    ], [conv[0], conv[1], conv[2]], option)
            else:
                heatmap, heatmap_gist_ncar = get_cam(i, averaged_gradients,
                                                     convmaps, option)
            if True:
                estimated_box = get_estimated_box(heatmap, option)
            else:
                estimated_box = get_estimated_box_nms_merge(heatmap, option)
            gt_box = get_gt_box(i, bbox)

            evaluator.accumulate_acc(estimated_box, gt_box, wrongs, i)
            bbox_img, concat = draw_images_with_boxes(i, images, heatmap,
                                                      estimated_box, gt_box)
            bbox_img, concat_gist_ncar = draw_images_with_boxes(
                i, images, heatmap_gist_ncar, estimated_box, gt_box)
            cls_name = meta[meta_labels[labels[i]]].split(',')[0]
            if False:
                get_estimated_box_draw_all(i, images, heatmap, option,
                                           cls_name)
            if False:
                save_3_branch_features(
                    i, images, [conv[0], conv[1], conv[2]], [
                        averaged_gradients_3x3, averaged_gradients_1x3,
                        averaged_gradients_3x1
                    ], averaged_gradients, convmaps, option, cls_name)

            if evaluator.cnt < 500:
                evaluator.save_img(threshold_idx, cls_name + "_hsv",
                                   concat_gist_ncar, option)
                # evaluator.save_img(threshold_idx, cls_name, concat, option)

            if evaluator.cnt == option.number_of_val:
                evaluator.compute_acc()
                evaluator.print_acc(threshold_idx, option)
                return


def evaluate_wsol(option, model, interval=False):
    if option.method_name in ['AAE', 'SAE', 'AAE_SAE']:
        eval_fun = evaluate_vgg16
    else:
        eval_fun = evaluate
    option.batch_size = 100
    if interval:
        for i in range(option.number_of_cam_curve_interval):
            option.cam_threshold = 0.05 + i * 0.05
            eval_fun(model, 'min-val-error-top1.index', option)
            # eval_fun(model, 'model-105000.index', option)
        if option.arch_name == 'resnet50_se':
            option.cam_threshold = 0.09
            # option.cam_threshold = 0.04
        else:
            # option.cam_threshold = 0.07
            option.cam_threshold = 0.09  # imagenet
        eval_fun(model, 'min-val-error-top1.index', option)
    else:
        # for i in range(7):
        #     option.cam_threshold = 0.07 + i * 0.01
        #     eval_fun(model, 'min-val-error-top1.index', option)
        if option.arch_name == 'resnet50_se':
            if option.dataset_name == 'CUB':
                option.cam_threshold = 0.09
            else:
                option.cam_threshold = 0.10
        else:
            if option.dataset_name == 'CUB':
                option.cam_threshold = 0.07
            else:
                option.cam_threshold = 0.09
        if option.method_name == 'CAM':
            option.cam_threshold = 0.2
        # option.cam_threshold = 0.09
        # for i in [0.08, 0.09, 0.1, 0.11]:
        #     option.cam_threshold = i
        eval_fun(model, 'min-val-error-top1.index', option)
