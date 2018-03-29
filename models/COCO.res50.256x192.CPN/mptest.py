import os
import os.path as osp
import numpy as np
import argparse
from config import cfg
import cv2
import sys
import time

import tensorflow as tf

from tfflat.base import Tester
from tfflat.utils import mem_info
from tfflat.logger import colorlogger
from network import Network

from lib_kernel.lib_nms.gpu_nms import gpu_nms
from lib_kernel.lib_nms.cpu_nms import cpu_soft_nms
from dataset import Preprocessing
from COCOAllJoints import COCOJoints

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

def test_net(tester, logger, dets, det_range):
    # here we assume all boxes are pre-processed.
    nms_method = 'nms'
    nms_thresh = 1.
    min_scores = 1e-10
    min_box_size = 0.  # 8 ** 2

    all_res = []
    dump_results = []

    start_time = time.time()

    img_start = det_range[0]
    while img_start < det_range[1]:
        img_end = img_start + 1
        im_info = dets[img_start]
        while img_end < det_range[1] and dets[img_end]['image_id'] == im_info['image_id']:
            img_end += 1

        test_data = dets[img_start:img_end]
        img_start = img_end

        iter_avg_cost_time = (time.time() - start_time) / (img_end - det_range[0])
        print('ran %.ds >> << left %.ds' % (
            iter_avg_cost_time * (img_end - det_range[0]), iter_avg_cost_time * (det_range[1] - img_end)))

        all_res.append([])

        # get box detections
        cls_dets = np.zeros((len(test_data), 5), dtype=np.float32)
        for i in range(len(test_data)):
            bbox = np.asarray(test_data[i]['bbox'])
            cls_dets[i, :4] = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            cls_dets[i, 4] = np.array(test_data[i]['score'])

        # nms and filter
        keep = np.where((cls_dets[:, 4] >= min_scores) &
                        ((cls_dets[:, 3] - cls_dets[:, 1]) * (cls_dets[:, 2] - cls_dets[:, 0]) >= min_box_size))[0]
        cls_dets = cls_dets[keep]
        if len(cls_dets) > 0:
            if nms_method == 'nms':
                keep = gpu_nms(cls_dets, nms_thresh)
            elif nms_method == 'soft':
                keep = cpu_soft_nms(np.ascontiguousarray(cls_dets, dtype=np.float32), method=2)
            else:
                assert False
        cls_dets = cls_dets[keep]
        test_data = np.asarray(test_data)[keep]

        if len(keep) == 0:
            continue

        # crop and detect keypoints
        cls_skeleton = np.zeros((len(test_data), cfg.nr_skeleton, 3))
        crops = np.zeros((len(test_data), 4))
        cfg.batch_size = 32
        batch_size = cfg.batch_size // 2
        for test_id in range(0, len(test_data), batch_size):
            start_id = test_id
            end_id = min(len(test_data), test_id + batch_size)

            test_imgs = []
            details = []
            for i in range(start_id, end_id):
                test_img, detail = Preprocessing(test_data[i], stage='test')
                test_imgs.append(test_img)
                details.append(detail)

            details = np.asarray(details)
            feed = test_imgs
            for i in range(end_id - start_id):
                ori_img = test_imgs[i][0].transpose(1, 2, 0)
                flip_img = cv2.flip(ori_img, 1)
                feed.append(flip_img.transpose(2, 0, 1)[np.newaxis, ...])
            feed = np.vstack(feed)

            res = tester.predict_one([feed.transpose(0, 2, 3, 1).astype(np.float32)])[0]
            res = res.transpose(0, 3, 1, 2)

            for i in range(end_id - start_id):
                fmp = res[end_id - start_id + i].transpose((1, 2, 0))
                fmp = cv2.flip(fmp, 1)
                fmp = list(fmp.transpose((2, 0, 1)))
                for (q, w) in cfg.symmetry:
                    fmp[q], fmp[w] = fmp[w], fmp[q]
                fmp = np.array(fmp)
                res[i] += fmp
                res[i] /= 2

            for test_image_id in range(start_id, end_id):
                r0 = res[test_image_id - start_id].copy()
                r0 /= 255.
                r0 += 0.5
                for w in range(cfg.nr_skeleton):
                    res[test_image_id - start_id, w] /= np.amax(res[test_image_id - start_id, w])
                border = 10
                dr = np.zeros((cfg.nr_skeleton, cfg.output_shape[0] + 2 * border, cfg.output_shape[1] + 2 * border))
                dr[:, border:-border, border:-border] = res[test_image_id - start_id][:cfg.nr_skeleton].copy()
                for w in range(cfg.nr_skeleton):
                    dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)
                for w in range(cfg.nr_skeleton):
                    lb = dr[w].argmax()
                    y, x = np.unravel_index(lb, dr[w].shape)
                    dr[w, y, x] = 0
                    lb = dr[w].argmax()
                    py, px = np.unravel_index(lb, dr[w].shape)
                    y -= border
                    x -= border
                    py -= border + y
                    px -= border + x
                    ln = (px ** 2 + py ** 2) ** 0.5
                    delta = 0.25
                    if ln > 1e-3:
                        x += delta * px / ln
                        y += delta * py / ln
                    x = max(0, min(x, cfg.output_shape[1] - 1))
                    y = max(0, min(y, cfg.output_shape[0] - 1))
                    cls_skeleton[test_image_id, w, :2] = (x * 4 + 2, y * 4 + 2)
                    cls_skeleton[test_image_id, w, 2] = r0[w, int(round(y) + 1e-10), int(round(x) + 1e-10)]
                # map back to original images
                crops[test_image_id, :] = details[test_image_id - start_id, :]
                for w in range(cfg.nr_skeleton):
                    cls_skeleton[test_image_id, w, 0] = cls_skeleton[test_image_id, w, 0] / cfg.data_shape[1] * (
                    crops[test_image_id][2] - crops[test_image_id][0]) + crops[test_image_id][0]
                    cls_skeleton[test_image_id, w, 1] = cls_skeleton[test_image_id, w, 1] / cfg.data_shape[0] * (
                    crops[test_image_id][3] - crops[test_image_id][1]) + crops[test_image_id][1]
        all_res[-1] = [cls_skeleton.copy(), cls_dets.copy()]

        cls_partsco = cls_skeleton[:, :, 2].copy().reshape(-1, cfg.nr_skeleton)
        cls_skeleton[:, :, 2] = 1
        cls_scores = cls_dets[:, -1].copy()

        # rescore
        cls_dets[:, -1] = cls_scores * cls_partsco.mean(axis=1)
        cls_skeleton = np.concatenate(
            [cls_skeleton.reshape(-1, cfg.nr_skeleton * 3), (cls_scores * cls_partsco.mean(axis=1))[:, np.newaxis]],
            axis=1)
        for i in range(len(cls_skeleton)):
            result = dict(image_id=im_info['image_id'], category_id=1, score=float(round(cls_skeleton[i][-1], 4)),
                          keypoints=cls_skeleton[i][:-1].round(3).tolist())
            dump_results.append(result)

    return all_res, dump_results


def test(test_model, logger):
    eval_gt = COCO(cfg.gt_path)
    import json
    with open(cfg.det_path, 'r') as f:
        dets = json.load(f)

    test_subset = False
    if test_subset:
        eval_gt.imgs = dict(list(eval_gt.imgs.items())[:100])
        anns = dict()
        for i in eval_gt.imgs:
            for j in eval_gt.getAnnIds(i):
                anns[j] = eval_gt.anns[j]
        eval_gt.anns = anns
    dets = [i for i in dets if i['image_id'] in eval_gt.imgs]

    dets = [i for i in dets if i['category_id'] == 1]
    dets.sort(key=lambda x: (x['image_id'], x['score']), reverse=True)
    for i in dets:
        i['imgpath'] = 'val2014/COCO_val2014_000000%06d.jpg' % i['image_id']
    img_num = len(np.unique([i['image_id'] for i in dets]))

    use_gtboxes = False
    if use_gtboxes:
        d = COCOJoints()
        coco_train_data, coco_test_data = d.load_data()
        coco_test_data.sort(key=lambda x: x['imgid'])
        for i in coco_test_data:
            i['image_id'] = i['imgid']
            i['score'] = 1.
        dets = coco_test_data

    from tfflat.mp_utils import MultiProc
    img_start = 0
    ranges = [0]
    images_per_gpu = int(img_num / len(args.gpu_ids.split(','))) + 1
    for run_img in range(img_num):
        img_end = img_start + 1
        while img_end < len(dets) and dets[img_end]['image_id'] == dets[img_start]['image_id']:
            img_end += 1
        if (run_img + 1) % images_per_gpu == 0 or (run_img + 1) == img_num:
            ranges.append(img_end)
        img_start = img_end

    def func(id):
        cfg.set_args(args.gpu_ids.split(',')[id])
        tester = Tester(Network(), cfg)
        tester.load_weights(test_model)
        range = [ranges[id], ranges[id + 1]]
        return test_net(tester, logger, dets, range)

    MultiGPUFunc = MultiProc(len(args.gpu_ids.split(',')), func)
    all_res, dump_results = MultiGPUFunc.work()

    # evaluation
    result_path = osp.join(cfg.output_dir, 'results.json')
    with open(result_path, 'w') as f:
        json.dump(dump_results, f)

    eval_dt = eval_gt.loadRes(result_path)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', '-d', type=str, dest='gpu_ids')
        parser.add_argument('--range', '-r', type=str, dest='test_epochs')
        parser.add_argument('--model', '-m', type=str, dest='test_model')
        args = parser.parse_args()

        # test gpus
        if not args.gpu_ids:
            args.gpu_ids = str(np.argmin(mem_info()))

        if '-' in args.gpu_ids:
            gpus = args.gpu_ids.split('-')
            gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
            gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
            args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

        if args.test_epochs and not ',' in args.test_epochs:
            args.test_epochs = '%d,%d' % (int(args.test_epochs), int(args.test_epochs) + 1)

        assert args.test_epochs or args.test_model, 'Test epoch or model to test is required.'
        return args

    global args
    args = parse_args()

    if args.test_model:
        logger = colorlogger(cfg.output_dir, 'test_model_{}'.format(args.test_model.split('/')[-1].split('.')[0]))
        test(args.test_model, logger)
    else:
        for i in range(*eval(args.test_epochs)):
            log_name = 'test_epoch_{}.logs'.format(i)
            logger = colorlogger(cfg.output_dir, log_name)
            test(i, logger)
