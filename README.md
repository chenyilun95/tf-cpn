# Cascaded Pyramid Network (CPN)

This is a Tensorflow re-implementation of CPN ([Cascaded Pyramid Network](https://arxiv.org/abs/1711.07319)), which wins 2017 COCO Keypoints Challenge. The original repo is based on the inner deep learning framework (MegBrain) in Megvii Inc.

## Results on COCO minival dataset (Single Model)
Note that our testing code is based on some detectors. In COCO minival dataset, the used detector here achieves an AP of 41.1 whose human AP is 55.3 in COCO minival dataset.
<center>

| Method | Base Model | Input Size | AP @0.5:0.95 | AP @0.5 | AP @0.75 | AP medium | AP large |
|:-------|:--------:|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|
| CPN | ResNet-50 | 256x192 | 69.7 | 88.3 | 77.0 | 66.2 | 76.1 |
| CPN | ResNet-50 | 384x288 | 72.3 | 89.1 | 78.8 | 68.4 | 79.1 |
| CPN | ResNet-101 | 384x288 | 72.9 | 89.2 | 79.4 | 69.1 | 79.9 | 

</center>

## Results on COCO test-dev dataset (Single Model)
Here we use the strong detector that achieves an AP of 44.5 whose human AP is 57.2 in COCO test-dev dataset.

<center>

| Method | AP @0.5:0.95 | AP @0.5 | AP @0.75 | AP medium | AP large |
|:-------|:-------:|:-------:|:-------:|:-------:|:-------:|
| Detectron(Mask R-CNN) | 67.0 | 88.0 | 73.1 | 62.2 | 75.6 |
| **CPN(ResNet-101, 384x288)** | **72.0** | **90.4** | **79.5** | **68.3** | **78.6** |

</center>

For reference, by using the detection results given by [MegDet](https://arxiv.org/abs/1711.07240) that achieves an AP of 52.1 whose human AP is 62.9, human pose result is as followed.

<center>

| Method | AP @0.5:0.95 | AP @0.5 | AP @0.75 | AP medium | AP large |
|:-------|:-------:|:-------:|:-------:|:-------:|:-------:|
| MegDet+CPN(ResNet-101, 384x288) | 73.0 | 91.8 | 80.8 | 69.1 | 78.7 |

</center>

## Usage

### Train on MSCOCO dataset
1. Clone the repository
```
git clone https://github.com/chenyilun95/tf-cpn.git
```
We'll call the directory that you cloned $CPN_ROOT.

2. Download MSCOCO images from [http://cocodataset.org/#download](http://cocodataset.org/#download). We train in COCO [trainvalminusminival](https://drive.google.com/drive/folders/15loPFQCMQnJqLK1viSMeIwTFT-KbNzdG?usp=sharing) dataset and validate in [minival](https://drive.google.com/drive/folders/15loPFQCMQnJqLK1viSMeIwTFT-KbNzdG?usp=sharing) dataset. Then put the data and evaluation [PythonAPI](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI) in $CPN_ROOT/data/COCO/MSCOCO. All paths are defined in config.py and you can modify them as you wish.

3. Download the base model (ResNet) weights from [slim model_zoo](https://github.com/tensorflow/models/tree/master/research/slim) and put them in $CPN_ROOT/data/imagenet_weights/. 

4. Setup your environment by first running
```
pip3 install -r requirement.txt
```

5. To train a CPN model, use network.py in the model folder.
```
python3 network.py -d 0-1
```
After the training finished, output is written underneath $CPN_ROOT/log/ which looks like below
```
log/
       |->model_dump/
       |    |->snapshot_1.ckpt.data-00000-of-00001
       |    |->snapshot_1.ckpt.index
       |    |->snapshot_1.ckpt.meta
       |    |->...
       |->train_logs.txt
```

### Validation
Run the testing code in the model folder. 
```
python3 mptest.py -d 0-1 -r 350
```
This assumes there is an models that has been trained for 350 epochs. If you just want to specify a pre-trained model path, it's fine to run
```
python3 mptest.py -d 0-1 -m log/model_dump/snapshot_350.ckpt
```

Here we provide the human detection boxes results:

[Person detection results in COCO Minival](https://drive.google.com/drive/folders/1BllF9--dN9uV3FRROcmuIbwNCcn7cCP0?usp=sharing)

[Person detection results in COCO test-dev](https://drive.google.com/open?id=1RNnWuLjLuqzemYzOjuoihQvLrBdwiVnu)

Pre-trained models:

[COCO.res50.256x192.CPN](https://drive.google.com/drive/folders/16hoq9KBVtR_zpJ1xUKweB-tbjFrao4sL?usp=sharing)

[COCO.res50.384x288.CPN](https://drive.google.com/drive/folders/1wP2agjJkDaBLl_1UcTmlmyl2Vw3CKlJa?usp=sharing)

[COCO.res101.384x288.CPN](https://drive.google.com/drive/folders/1X0kcPG1KSn3aeWm9fvqVMziGK3XIvJv9?usp=sharing)

## Citing CPN
If you find [CPN](https://arxiv.org/abs/1711.07319) useful in your research, please consider citing:

    @article{Chen2018CPN,
        Author = {Chen, Yilun and Wang, Zhicheng and Peng, Yuxiang and Zhang, Zhiqiang and Yu, Gang and Sun, Jian},
        Title = {{Cascaded Pyramid Network for Multi-Person Pose Estimation}},
        Conference = {CVPR},
        Year = {2018}
    }

## Third party implementation
Thanks for [Geng David](https://github.com/GengDavid) and his [pytorch re-implementation of CPN](https://github.com/GengDavid/pytorch-cpn).

## Troubleshooting
1. If you find it pending while running mptest.py, it may be the blocking problem of python queue in multiprocessing. For convenience, I simply implemented data transferring via temporary files. You need to call MultiProc with extra parameter "dump_method=1" and it'll be fine to run the test code with multiprocess.

## Contact
If you have any questions about this repo, please feel free to contact chenyilun95@gmail.com.
