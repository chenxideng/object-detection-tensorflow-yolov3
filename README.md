## Part 1. Introduction [[Tutorial]](https://github.com/YunYang1994/ai-notebooks/blob/master/YOLOv3.md)

Implementation of YOLO v3 object detector in Tensorflow. The full details are in [this paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).  In this project we cover several segments as follows:<br>
- [x] [YOLO v3 architecture]
- [x] [Training tensorflow-yolov3 with GIOU loss function](https://giou.stanford.edu/)
- [x] Basic working demo
- [x] Training pipeline
- [x] Multi-scale training method
- [x] Compute VOC mAP

YOLO paper is quick hard to understand, along side that paper. This repo enables you to have a quick understanding of YOLO Algorithmn.


## Part 2. Quick start
1. Clone this file
```bashrc
$ sudo git clone --branch <branchname> https://github.com/chenxideng/object-detection-tensorflow-yolov3.git
```

2.  You are supposed to install some dependencies before getting out hands with these codes s.t. CUDA 10.0, Tensorflow-gpu 1.14.0, & cuDNN 7.5.0. please follow steps to install CUDA & cuDNN: 
https://medium.com/repro-repo/install-cuda-10-1-and-cudnn-7-5-0-for-pytorch-on-ubuntu-18-04-lts-9b6124c44cc
```bashrc
$ cd object-detection-tensorflow-yolov3
$ sudo pip3 install -r ./docs/requirements.txt
```
check cuda version and check if more than one cuda version installed, just erase lower version and keep ONE highest version.
```bashrc
$ sudo nvidia-smi
$ cd /use/local
```

3. Exporting loaded COCO weights as TF checkpoint(`yolov3_coco.ckpt`)【[BaiduCloud](Link: https://pan.baidu.com/s/1YjuM0VcAm0MTRMDH5LL3iw Code: 7die)】
```bashrc
$ cd checkpoint
$ sudo wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
$ sudo tar -xvf yolov3_coco.tar.gz
$ cd ..
$ sudo python3 convert_weight.py
$ sudo python3 freeze_graph.py
```
4. Then you will get some `.pb` files in the root path.,  and run the demo script
```bashrc
$ sudo python3 image_demo.py
$ sudo python3 video_demo.py --input <video_path> # if use camera, set video_path = 0
```

## Part 3. Train your own dataset
Two files are required as follows:

- [`dataset.txt`](https://raw.githubusercontent.com/YunYang1994/tensorflow-yolov3/master/data/dataset/voc_train.txt): 

```
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# make sure that x_max < width and y_max < height
```

- [`class.names`](https://github.com/YunYang1994/tensorflow-yolov3/blob/master/data/classes/coco.names):

```
person
bicycle
car
...
toothbrush
```
if you add classes, please create a new file

### 3.1 Train VOC dataset
To help you understand my training process, I made this demo of training VOC PASCAL dataset
#### how to train it ?
Download VOC PASCAL trainval  and test data
```bashrc
$ sudo wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ sudo wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ sudo wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
Extract all of these tars into one directory and rename them, which should have the following basic structure.

```bashrc

VOC           # path:  /home/charles/object-detection/VOC
├── test
|    └──VOCdevkit
|        └──VOC2007 (from VOCtest_06-Nov-2007.tar)
└── train
     └──VOCdevkit
         └──VOC2007 (from VOCtrainval_06-Nov-2007.tar)
         └──VOC2012 (from VOCtrainval_11-May-2012.tar)
                     
$ sudo python3 scripts/voc_annotation.py --data_path /home/charles/object-detection/VOC
```
Then edit your `./core/config.py` to make some necessary configurations

```bashrc
__C.YOLO.CLASSES                = "./data/classes/voc.names"
__C.TRAIN.ANNOT_PATH            = "./data/dataset/voc_train.txt"
__C.TEST.ANNOT_PATH             = "./data/dataset/voc_test.txt"
```
Here are two kinds of training method: 

##### (1) train from scratch:

```bashrc
$ sudo python3 train.py
```
##### Launch tensorboard
```bashrc
$ sudo python3 -m tensorboard.main --logdir=data
```
##### (2) train from COCO weights(recommend):

```bashrc
$ cd checkpoint
$ wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
$ sudo tar -xvf yolov3_coco.tar.gz
$ cd ..
$ sudo python3 convert_weight.py --train_from_coco
$ sudo python3 train.py
```
##### (3) train from your own datasets:
##### Label from LabelImg (an open source for labeling)
1. Create classes and datasets txt file (id from 0);
2. Modify classes number of demo.py to match the number of classes;
3. Run train.py
4. After training, update core/config.py to use the new weight file located in checkpoints/, and update classes txt;
5. Update freeze.py file to use the right weight and create *.pb, run freeze.py to generate *.pb file;
6. Update demo.py to use the new *.pb file.

#### how to test and evaluate it ?
```
$ sudo python3 evaluate.py
$ cd mAP
$ sudo python3 main.py -na
```
if you are still unfamiliar with training pipline, you can join [here](https://github.com/YunYang1994/tensorflow-yolov3/issues/39) to discuss with us.

### 3.2 Train other dataset
Download COCO and test data
```
$ sudo wget http://images.cocodataset.org/zips/train2017.zip
$ sudo wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ sudo wget http://images.cocodataset.org/zips/test2017.zip
$ sudo wget http://images.cocodataset.org/annotations/image_info_test2017.zip 
```

## Part 4. Other Implementations


[-**`Stronger-yolo`**](https://github.com/Stinky-Tofu/Stronger-yolo)<br>

[- **`Implementing YOLO v3 in Tensorflow (TF-Slim)`**](https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe)

[- **`YOLOv3_TensorFlow`**](https://github.com/wizyoung/YOLOv3_TensorFlow)

[- **`Object Detection using YOLOv2 on Pascal VOC2012`**](https://fairyonice.github.io/Part_1_Object_Detection_with_Yolo_for_VOC_2014_data_anchor_box_clustering.html)

[-**`Understanding YOLO`**](https://hackernoon.com/understanding-yolo-f5a74bbc7967)

