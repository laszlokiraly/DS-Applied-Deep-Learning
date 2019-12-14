# Journal

## Cuda

Existing Hard/Software: **Intel i7 2600K @3.40GHz, 24GB RAM, GTX-1080-TI 11GB RAM**  
Check existing pytorch and cuda installation for functionality and version: **pytorch version: 1.1.0, cuda tests OK**

- 0.5h setup cuda

## HR-net

### Setup

<https://github.com/HRNet/HRNet-Image-Classification> includes requirements and setup process.  
Required pytorch version is 0.4.1.

#### Problems

When installing requirements there were problems with Shapely 1.6.4 on my windows 10 machine:  
- https://stackoverflow.com/questions/35991403/pip-install-unroll-python-setup-py-egg-info-failed-with-error-code-1#36025294
- (https://stackoverflow.com/questions/43199480/esay-install-u-setuptools-failing-with-error-winerror-5-access-is-denied-c)
- https://stackoverflow.com/questions/44398265/install-shapely-oserror-winerror-126-the-specified-module-could-not-be-found

#### Solution

-> download Shapely from https://pypi.org/project/Shapely/#files and install via wheel

```bash
$ python -m pip install Shapely-1.6.4.post2-cp36-cp36m-win_amd64.whl
[...]
```

and uncomment # shapely==1.6.4 in requirements.txt

### Validation

Before starting the pipeline I wanted to test that model and cuda are working correctly for prediction of imagenet validation dataset.

Used cls hrnet w18 pretrained model from onedrive as described in https://github.com/HRNet/HRNet-Image-Classification.

After

- hard coding one gpu instead of four
- fiddling with .cuda(gpu_device)
- and adding logging of dataset size and progress dots

it finally worked and this is the result of imagenet classification validation for hrnet w18 model:  

```bash
$ python tools/valid.py --cfg experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --testModel hrnetv2_w18_imagenet_pretrained.pth --dataDir=../../data
=> creating output\imagenet\cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100
=> creating log\imagenet\cls_hrnet\cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100_2019-12-07-20-51
Namespace(cfg='experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml', dataDir='../../data', logDir='', modelDir='', testModel='hrnetv2_w18_imagenet_pretrained.pth')
{'AUTO_RESUME': False,
 'CUDNN': CfgNode({'BENCHMARK': True, 'DETERMINISTIC': False, 'ENABLED': True}),
 'DATASET': {'COLOR_RGB': False,
             'DATASET': 'imagenet',
             'DATA_FORMAT': 'jpg',
             'FLIP': True,
             'HYBRID_JOINTS_TYPE': '',
             'NUM_JOINTS_HALF_BODY': 8,
             'PROB_HALF_BODY': 0.0,
             'ROOT': '../data\\imagenet\\images',
             'ROT_FACTOR': 30,
             'SCALE_FACTOR': 0.25,
             'SELECT_DATA': False,
             'TEST_SET': 'val',
             'TRAIN_SET': 'train'},
 'DATA_DIR': '../data',
 'DEBUG': {'DEBUG': False,
           'SAVE_BATCH_IMAGES_GT': False,
           'SAVE_BATCH_IMAGES_PRED': False,
           'SAVE_HEATMAPS_GT': False,
           'SAVE_HEATMAPS_PRED': False},
 'GPUS': (0, 1, 2, 3),
 'LOG_DIR': 'log/',
 'LOSS': {'TOPK': 8,
          'USE_DIFFERENT_JOINTS_WEIGHT': False,
          'USE_OHKM': False,
          'USE_TARGET_WEIGHT': True},
 'MODEL': {'EXTRA': {'STAGE1': {'BLOCK': 'BOTTLENECK',
                                'FUSE_METHOD': 'SUM',
                                'NUM_BLOCKS': [4],
                                'NUM_CHANNELS': [64],
                                'NUM_MODULES': 1,
                                'NUM_RANCHES': 1},
                     'STAGE2': {'BLOCK': 'BASIC',
                                'FUSE_METHOD': 'SUM',
                                'NUM_BLOCKS': [4, 4],
                                'NUM_BRANCHES': 2,
                                'NUM_CHANNELS': [18, 36],
                                'NUM_MODULES': 1},
                     'STAGE3': {'BLOCK': 'BASIC',
                                'FUSE_METHOD': 'SUM',
                                'NUM_BLOCKS': [4, 4, 4],
                                'NUM_BRANCHES': 3,
                                'NUM_CHANNELS': [18, 36, 72],
                                'NUM_MODULES': 4},
                     'STAGE4': {'BLOCK': 'BASIC',
                                'FUSE_METHOD': 'SUM',
                                'NUM_BLOCKS': [4, 4, 4, 4],
                                'NUM_BRANCHES': 4,
                                'NUM_CHANNELS': [18, 36, 72, 144],
                                'NUM_MODULES': 3}},
           'HEATMAP_SIZE': [64, 64],
           'IMAGE_SIZE': [224, 224],
           'INIT_WEIGHTS': True,
           'NAME': 'cls_hrnet',
           'NUM_CLASSES': 1000,
           'NUM_JOINTS': 17,
           'PRETRAINED': '',
           'SIGMA': 2,
           'TAG_PER_JOINT': True,
           'TARGET_TYPE': 'gaussian'},
 'OUTPUT_DIR': 'output/',
 'PIN_MEMORY': True,
 'PRINT_FREQ': 1000,
 'RANK': 0,
 'TEST': {'BATCH_SIZE_PER_GPU': 32,
          'BBOX_THRE': 1.0,
          'COCO_BBOX_FILE': '',
          'FLIP_TEST': False,
          'IMAGE_THRE': 0.1,
          'IN_VIS_THRE': 0.0,
          'MODEL_FILE': 'hrnetv2_w18_imagenet_pretrained.pth',
          'NMS_THRE': 0.6,
          'OKS_THRE': 0.5,
          'POST_PROCESS': False,
          'SHIFT_HEATMAP': False,
          'SOFT_NMS': False,
          'USE_GT_BBOX': False},
 'TRAIN': {'BATCH_SIZE_PER_GPU': 32,
           'BEGIN_EPOCH': 0,
           'CHECKPOINT': '',
           'END_EPOCH': 100,
           'GAMMA1': 0.99,
           'GAMMA2': 0.0,
           'LR': 0.05,
           'LR_FACTOR': 0.1,
           'LR_STEP': [30, 60, 90],
           'MOMENTUM': 0.9,
           'NESTEROV': True,
           'OPTIMIZER': 'sgd',
           'RESUME': True,
           'SHUFFLE': True,
           'WD': 0.0001},
 'WORKERS': 1}
=> init weights from normal distribution

Total Parameters: 21,299,004
----------------------------------------------------------------------------------------------------------------------------------
Total Multiply Adds (For Convolution and Linear Layers only): 3.9893547743558884 GFLOPs
----------------------------------------------------------------------------------------------------------------------------------
Number of Layers
Conv2d : 325 layers   BatchNorm2d : 325 layers   ReLU : 284 layers   Bottleneck : 8 layers   BasicBlock : 104 layers   Upsample : 31 layers   HighResolutionModule : 8 layers   Linear : 1 layers
=> loading model from hrnetv2_w18_imagenet_pretrained.pth
number of images: 50000
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
.................................................o
............
Test: Time 0.791    Loss 0.9106     Error@1 23.244  Error@5 6.558   Accuracy@1 76.756       Accuracy@5 93.442
```

- 1h hrnet overview
- 5h hrnet imagenet validation of w18 model (in /hrnet-imagenet-valid)

## setting up transfer learning example

<https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html> links to interesting transfer learning article <https://cs231n.github.io/transfer-learning/>

### problems

#### pil library

```bash
UnboundLocalError: local variable 'photoshop' referenced before assignment
```

Pillow was 6.0.0, upgrade to 6.2.1 helped solve the issue.

```bash
$ pip uninstall Pillow
[...]
$ pip install Pillow
[...]
```

- 2.25h

## unit test (wip)

using [torch test](https://github.com/suriyadeepan/torchtest)

- 0.5h

## setup transfer learning with heritage dataset

### train dataset problem

the train dataset only contains 128x128 images, the train dataset with different images sizes is blocked by an authorization login. Wrote email to joslla@cartif.es

- 3h

## TODO

### mandadory

- put data onto ssd to improve train time
- f1 score
- save model
- application
  - load model
  - prediction for an image
  - web client (simple upload button in html/css)
  - web server (simple json response in flask)
  - dockerize

### optional

- use hrnet
  - f1 score
  - add to application:
    - web server
    - prediction
    - dockerize
