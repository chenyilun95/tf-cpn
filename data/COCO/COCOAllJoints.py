#!/usr/bin/python3
# coding=utf-8

import os
import os.path as osp
import numpy as np
import cv2

import sys
cur_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_dir, 'MSCOCO', 'PythonAPI'))
from pycocotools.coco import COCO

class COCOJoints(object):
    def __init__(self):
        self.kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',
        'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
        'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
        self.max_num_joints = 17
        self.color = np.random.randint(0, 256, (self.max_num_joints, 3))

        self.mpi = []
        self.test_mpi = []
        for mpi, stage in zip([self.mpi, self.test_mpi], ['train', 'val']):
            if stage == 'train':
                self._train_gt_path=os.path.join(cur_dir, 'MSCOCO', 'annotations', 'person_keypoints_trainvalminusminival2014.json')
                coco = COCO(self._train_gt_path)
            else:
                self._val_gt_path=os.path.join(cur_dir, 'MSCOCO', 'annotations', 'person_keypoints_minival2014.json')
                coco = COCO(self._val_gt_path)
            if stage == 'train':
                for aid in coco.anns.keys():
                    ann = coco.anns[aid]
                    if ann['image_id'] not in coco.imgs or ann['image_id'] == '366379':
                        continue
                    imgname = coco.imgs[ann['image_id']]['file_name']
                    prefix = 'val' if 'val' in imgname else 'train'
                    rect = np.array([0, 0, 1, 1], np.int32)
                    if ann['iscrowd']:
                        continue
                    joints = ann['keypoints']
                    bbox = ann['bbox']
                    if np.sum(joints[2::3]) == 0 or ann['num_keypoints'] == 0 :
                        continue
                    imgname = prefix + '2014/' + 'COCO_' + prefix + '2014' + '_' + str(ann['image_id']).zfill(12) + '.jpg'
                    humanData = dict(aid = aid,joints=joints, imgpath=imgname, headRect=rect, bbox=bbox, imgid = ann['image_id'], segmentation = ann['segmentation'])
                    mpi.append(humanData)
            elif stage == 'val':
                files = [(img_id,coco.imgs[img_id]) for img_id in coco.imgs]
                for img_id,img_info in files:
                    imgname = stage + '2014/' + img_info['file_name']
                    humanData = dict(imgid = img_id,imgpath = imgname)
                    mpi.append(humanData)
            else:
                print('COCO data error, please check')
                embed()

    def load_data(self, min_kps=1):
        mpi = [i for i in self.mpi if np.sum(np.array(i['joints'], copy=False)[2::3] > 0) >= min_kps]
        return mpi, self.test_mpi

if __name__ == '__main__':
    coco_joints = COCOJoints()
    train, test = coco_joints.load_data(min_kps=1)
    from IPython import embed; embed()
