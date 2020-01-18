# Applied Deep Learning: Exercise 2 - Hacking


Author: **"Laszlo Kiraly, 09227679"**  
Date: **"18/12/2019"**

---

## Project Idea
The paper [Classification of Architectural Heritage Images Using Deep Learning Techniques](https://www.researchgate.net/publication/320052364_Classification_of_Architectural_Heritage_Images_Using_Deep_Learning_Techniques) brings its own [dataset](https://old.datahub.io/dataset/architectural-heritage-elements-image-dataset) to classify different types of architectural monuments, e.g. gargoyles, inner and outer domes, altars, etc.

The scores for full retrain of a resnet model are:

| Measure | Altar | Apse | Bell Tower | Column | Dome Inner | Dome Outer | Flying Buttress | Gargoyle | Stained Glass | Vault |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 score | 0.906 | 0.874 | 0.903 | 0.953 | 0.967 | 0.937 | 0.805 | 0.923 | 0.990 | 0.925 |

![scores and training time from paper](./scores-paper.png)

The goal of my project is to beat the F1 Score on at least 6 classes with transfer learning.

## Dataset

The [dataset](https://old.datahub.io/dataset/architectural-heritage-elements-image-dataset) has 10 classes of architectural heritages.
There are 10235 images available from which 80% were taken for training and 20% for validation. The test dataset consists of 1404 elements.
The images dimensions are 128x128 with 8bit RBG in jpg format.

## Result Table

Four experiments of full transfer learning with 2 different model families have been concluded.

**model**|**altar**|**apse**|**bell tower**|**column**|**dome(inner)**|**dome(outer)**|**flying buttress**|**gargoyle**|**stained glass**|**vault**|**performance**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
**baseline**      |0.906     |0.874|0.903 |0.953|0.967|0.937 |0.805 |0.923 |0.990 |0.925|+/-0
**hrnet v1 small**|**0.9353**|0.713|0.8521|0.866|0.922|0.9043|**0.8442**|**0.9437**|0.9728|**0.9254**|-2
**resnet 18**     |**0.942**|0.6602|0.8629|0.8427|0.9078|0.8762|0.766|**0.9536**|**0.9622**|**0.9298**|-2
**resnet 152**    |**0.96**|0.8738|**0.9193**|0.9448|0.9014|0.9356|**0.9262**|**0.9853**|0.9547|**0.9446**|+/-0
**hrnet v2 largest**|**0.9373**|**0.9091**|**0.944**|0.9458|0.9489|0.936|**0.9396**|**0.9787**|0.966|**0.9364**|**+2**
**hrnet v2 largest II**|**0.9407**|**0.8889**|0.9003|**0.9369**|0.9051|0.9103|**0.9589**|**0.9895**|0.9796|**0.9443**|**+2**

*hrnet v2* beats the baseline model in six of ten classes and the f1 score is remarkable well balanced over all classes. The baseline f1 scores `max - min` is 0.185, whereas *hrnet v2* max distance of the classes is 0.0696. Remark: The procedure of saving the transferred high resolution net model has changed, so a second model (`hrnet v2 largest II`) with same parameters was trained and added to the comparison.

# Time tracking

### Planned Times

Work Breakdown

- setup of current cuda environment @home gtx-1080ti (4h)
- check if there is a more current deep neural network trained on images which can be used for transfer learning, if there isn't a usable one use then use one from the paper, e.g. inception v3 (4h)
- get inspiration from [A Survey on Deep Transfer Learning](https://link.springer.com/chapter/10.1007/978-3-030-01424-7_27) and/or [Hands-On Transfer Learning with Python](https://proquest.tech.safaribooksonline.de/9781788831307) and/or [fast.ai](https://www.fast.ai) and/or [tensorflow](https://www.tensorflow.org/hub/tutorials/image_retraining) AND [HRNet]( https://github.com/HRNet/HRNet-Image-Classification)(8h)
- network architecture first draft including F1 score results (8h)
- train, evaluate and fine-tune the network (16h)
- TODO Exercise 3: web application for uploading and classifying an image with the latest model in a docker image (4h)
- TODO Exercise 3: final report (4h)
- TODO Exercise 3: presentation slides (4h)


### Real Times

Work Breakdown

- setup environment (cuda & transfer example) (0.5h)
- hrnet overview and validation of pretrained model (6h)
- dummy torch test setup (0.5h)
- adapt pytorch transfer learning tutorial for heritage dataset (12h)
- include hrnet model into transfer learning for heritage dataset with F1-Score (8h)
- run experiments on all models and evaluate/document results (2h)
- refactor code to indidual files (1.5h)
- package README.md (0.5h)
- prediction of a single image (4.5h)

=> Total Time: 35.5

## setup

### conda environment

gpu:

```bash
conda create -n pytorch_gpu
conda activate pytorch_gpu
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install yacs
conda install matplotlib
conda install scikit-learn
conda install tabulate
conda install dill
conda install pillow=6.1
pip install torchtest
```

cpu:

```bash
conda create -n pytorch_gpu
conda activate pytorch_gpu
conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch
pip install yacs
conda install matplotlib
conda install scikit-learn
conda install tabulate
conda install dill
conda install pillow=6.1
pip install torchtest
```

Put the pretrained hrnet models from [hrnet github page](https://github.com/HRNet/HRNet-Image-Classification#imagenet-pretrained-models) into the `transfer/`

The [heritage dataset](https://old.datahub.io/dataset/architectural-heritage-elements-image-dataset) folder can be configured, the `val` folder has to be renamed to `test`.

## run code

Main entry point is `transfer/train.py`.  

The yaml files, which are needed for hrnet only, hold configurations of data folders, model architecture, batch size, hyper parameters, etc. and are found in `transfer/experiments`.

Additional parameters are defined as extra arguments:
- modelFamily: one of resnet18 resnet152, hrnet
- transferEpochs: number of epochs for transfer learning
- transferBatchSize: batch size for transfer learning
- transferDataFolder: folder where the [heritage dataset](https://old.datahub.io/dataset/architectural-heritage-elements-image-dataset) resides in

see `transfer/run.sh` for four usages of transfer learning, which have been used for training and testing.

## next steps

### deployment
- https://medium.com/datadriveninvestor/deploy-your-pytorch-model-to-production-f69460192217
- https://github.com/alecrubin/pytorch-serverless/
- ...

### optimization
- prune model?
