import matplotlib.pyplot as plt
import random
import numpy as np
import cv2

def visualize(img, det_boxes=None, gt_boxes=None, keypoints=None, is_show_label=True, show_cls_label = True, show_skeleton_labels=False, classes=None, thresh=0.5, name='detection', return_img=False):
    if is_show_label:
        if classes == 'voc':
            classes = [
                '__background__', 
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor'
            ]
        elif classes == 'coco':
            classes = [
                "__background__",
                "person", "bicycle", "car", "motorcycle", "airplane", 
                "bus", "train", "truck", "boat", "traffic light", 
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", 
                "cat", "dog", "horse", "sheep", "cow", 
                "elephant", "bear", "zebra", "giraffe", "backpack", 
                "umbrella", "handbag", "tie", "suitcase", "frisbee", 
                "skis", "snowboard", "sports ball", "kite", "baseball bat", 
                "baseball glove", "skateboard", "surfboard", "tennis racket","bottle", 
                "wine glass", "cup", "fork", "knife", "spoon", 
                "bowl", "banana", "apple", "sandwich", "orange", 
                "broccoli", "carrot", "hot dog", "pizza", "donut", 
                "cake", "chair", "couch", "potted plant", "bed", 
                "dining table", "toilet", "tv", "laptop", "mouse", 
                "remote", "keyboard", "cell phone", "microwave", "oven", 
                "toaster", "sink", "refrigerator", "book", "clock", 
                "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            ]


    color_map = [(0, 0, 0), (0, 255, 0), (255, 128, 0), (255, 255, 0), (255, 0, 255), (255, 128, 255), (128, 255, 128), (128, 255, 255), (255, 255, 128), (0, 128, 255), (0, 255, 128),
            (255, 0, 128), (0, 215, 255), (255, 0, 255), (255, 128, 0), (128, 128, 255), (0, 255, 255), (0, 69, 255), (0, 69, 255), (255, 204, 204), (204, 255, 255)]

    im = np.array(img).copy().astype(np.uint8)
    colors = dict()
    font = cv2.FONT_HERSHEY_SIMPLEX


    if det_boxes is not None:
        det_boxes = np.array(det_boxes)
        for det in det_boxes:
            bb = det[:4].astype(int)
            if is_show_label:
                if show_cls_label:
                    cls_id = int(det[4])
                    if cls_id == 0:
                        continue
                if len(det) > 4:
                    score = det[-1]
                else:
                    score = 1.
                if thresh < score:
                    if show_cls_label:
                        if cls_id not in colors:
                            colors[cls_id] = (random.random() * 128 + 128, random.random() * 128 + 128, random.random() * 128 + 128)
                        cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), colors[cls_id], 1)

                        if classes and len(classes) > cls_id:
                            cls_name = classes[cls_id]
                        else:
                            cls_name = str(cls_id)
                        cv2.putText(im, '{:s} {:.3f}'.format(cls_name, score), (bb[0], bb[1] - 2), font, 0.7, colors[cls_id], 2)
                    else:
                        cv2.putText(im, '{:.3f}'.format(score), (bb[0], bb[1] - 2), font, 0.7, (255, 0, 0), 2)
                        cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (139, 139, 139), 1)
            else:
                cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (random.random() * 128 + 128, random.random() * 128 + 128, random.random() * 128 + 128), 1)

    if gt_boxes is not None:
        gt_boxes = np.array(gt_boxes)
        for gt in gt_boxes:
            bb = gt[:4].astype(int)
            if is_show_label:
                cls_id = int(gt[4])
                cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 3)

                if classes and len(classes) > cls_id:
                    cls_name = classes[cls_id]
                else:
                    cls_name = str(cls_id)
                cv2.putText(im, '{:s}'.format(cls_name), (bb[0], bb[1] - 2), \
                            font, 0.5, (0, 0, 255), 1)
            else:
                cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 3)

    if keypoints is not None:
        keypoints = np.array(keypoints).astype(int)
        keypoints = keypoints.reshape(-1, 17, 3)

        if False:
            idx = np.where(det_boxes[:, -1] > thresh)
            keypoints = keypoints[idx]
            for i in range(len(keypoints)):
                draw_skeleton(im, keypoints[i], show_skeleton_labels)
        else:
            for i in range(len(keypoints)):
                draw_skeleton(im, keypoints[i], show_skeleton_labels)

    if return_img:
        return im.copy()

    cv2.imshow(name, im)
    cv2.waitKey(0)
    # while True:
    #     c = cv2.waitKey(0)
    #     if c == ord('d'):
    #         return
    #     elif c == ord('n'):
    #         break

def draw_skeleton(aa, kp, show_skeleton_labels=False):
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 
    'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 
    'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']

    for i, j in skeleton:
        if kp[i-1][0] >= 0 and kp[i-1][1] >= 0 and kp[j-1][0] >= 0 and kp[j-1][1] >= 0 and \
            (len(kp[i-1]) <= 2 or (len(kp[i-1]) > 2 and  kp[i-1][2] > 0.1 and kp[j-1][2] > 0.1)):
            cv2.line(aa, tuple(kp[i-1][:2]), tuple(kp[j-1][:2]), (0,255,255), 2)
    for j in range(len(kp)):
        if kp[j][0] >= 0 and kp[j][1] >= 0:

            if len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 1.1):
                cv2.circle(aa, tuple(kp[j][:2]), 2, tuple((0,0,255)), 2)
            elif len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1):
                cv2.circle(aa, tuple(kp[j][:2]), 2, tuple((255,0,0)), 2)

            if show_skeleton_labels and (len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1)):
                cv2.putText(aa, kp_names[j], tuple(kp[j][:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
