# Applied Deep Learning: Report of Improving on Classification of Architectural Heritage Images Using Deep Learning Techniques

Author: **"Laszlo Kiraly, 09227679"**  
Date: **"20/01/2020"**

---

## Project Idea

The paper [Classification of Architectural Heritage Images Using Deep Learning Techniques](https://www.researchgate.net/publication/320052364_Classification_of_Architectural_Heritage_Images_Using_Deep_Learning_Techniques) brings its own [dataset](https://old.datahub.io/dataset/architectural-heritage-elements-image-dataset) to classify different types of architectural monuments, e.g. gargoyles, inner and outer domes, altars, etc.

The scores for full retrain of a resnet model are:

| Measure | Altar | Apse | Bell Tower | Column | Dome Inner | Dome Outer | Flying Buttress | Gargoyle | Stained Glass | Vault |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 score | 0.906 | 0.874 | 0.903 | 0.953 | 0.967 | 0.937 | 0.805 | 0.923 | 0.990 | 0.925 |

![scores and training time from paper](./scores-paper.png)

The goal of my project is to beat the F1 Score on at least 6 classes with transfer learning.  
A simple containerized web app for uploading and predicting of images is also provided.

## Dataset

The [dataset](https://old.datahub.io/dataset/architectural-heritage-elements-image-dataset) has 10 classes of architectural heritages.
There are 10235 images available from which 80% were taken for training and 20% for validation. The test dataset consists of 1404 elements.
The training images dimensions are 128x128 with 8bit RBG in jpg format. The test images are 64x64 in size.

## Result Table

Four experiments of full transfer learning with 2 different model families have been concluded. The models have been trained in 25 epochs:

**model**|**altar**|**apse**|**bell tower**|**column**|**dome(inner)**|**dome(outer)**|**flying buttress**|**gargoyle**|**stained glass**|**vault**|**performance**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
**baseline**           |0.906     |0.874     |0.903    |0.953 |0.967|0.937 |0.805 |0.923 |0.990 |0.925|+/-0
**hrnet v1 small**     |**0.9353**|0.713|0.8521|0.866|0.922|0.9043|**0.8442**|**0.9437**|0.9728|**0.9254**|-2
**resnet 18**          |**0.942**|0.6602|0.8629|0.8427|0.9078|0.8762|0.766|**0.9536**|**0.9622**|**0.9298**|-2
**resnet 152**         |**0.96**|0.8738|**0.9193**|0.9448|0.9014|0.9356|**0.9262**|**0.9853**|0.9547|**0.9446**|+/-0
**hrnet v2 largest**   |**0.9373**|**0.9091**|**0.944**|0.9458|0.9489|0.936|**0.9396**|**0.9787**|0.966|**0.9364**|+2

`hrnet v2 largest` beats the `baseline` model in six of ten classes and the f1 score is remarkable well balanced over all classes. The baseline f1 scores interval length of `max - min` is 0.185, whereas `hrnet v2 largest` max/min distance of the classes is 0.0696.  
After problems with saving/loading the transferred models, resnet and hrnet have been retrained with the same parameters, but only in 15 epochs:

**model**|**altar**|**apse**|**bell tower**|**column**|**dome(inner)**|**dome(outer)**|**flying buttress**|**gargoyle**|**stained glass**|**vault**|**performance**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
**resnet 152**      |**0.9438**|**0.8958**|**0.908**|0.9343|0.913|0.9158|**0.9524**|**0.9874**|0.9831|**0.9474**|+2
**hrnet v2 largest**|**0.9412**|**0.8842**|**0.9373**|0.9474|0.942|**0.9488**|**0.9272**|**0.9809**|0.9655|**0.9275**|**+4**

Nice!

## Setup

### Python Dependencies

The dependencies include all libraries needed for validating hrnet, transfer learning with resnet and hrnet, as well as a simple flask web app.

```bash
pip install -r docker-pytorch-base/requirements.txt
```

For gpu based transfer learning also add cuda via

```bash
pip install cudatoolkit=10.0
```

### Pretrained Models

Put the pretrained hrnet models linked at [hrnet github page](https://github.com/HRNet/HRNet-Image-Classification#imagenet-pretrained-models) into the folder `transfer/model_states`.

The resnet models will be automatically downloaded at training time.

### Heritage Dataset

The [heritage dataset](https://old.datahub.io/dataset/architectural-heritage-elements-image-dataset) target folder can be configured, but the `val` subfolder has to be renamed to `test`.

## Transfer the models

Main entry point is `transfer/train.py`.  

The yaml files, which are needed for hrnet only, hold configurations of data folders, model architecture, batch size, hyper parameters, etc. and are found in `transfer/experiments`.

Additional parameters are defined as extra arguments:

- modelFamily: one of resnet18 resnet152, hrnet
- transferEpochs: number of epochs for transfer learning
- transferBatchSize: batch size for transfer learning
- transferDataFolder: folder where the [heritage dataset](https://old.datahub.io/dataset/architectural-heritage-elements-image-dataset) resides in

example:

```python
python train.py --cfg=experiments/cls_resnet152.yaml --dataDir=../../data --modelFamily=resnet152 --transferEpochs=25 --transferBatchSize=32 > resnet152-transfer-epochs25.log
```

see `transfer/run.sh` for four usages of transfer learning, which have been used for training and testing.

## Prepare Transferred Models

If you have transferred the models yoursel in previous steps, then there should be `hrnet_final_model.pt` and `resnet152_final_model.pt` in the `transfer` folder.  
Otherwise you can download the pytorch models from [https://github.com/laszlokiraly/DS-Applied-Deep-Learning/releases/tag/v1.0.0-rc.1](https://github.com/laszlokiraly/DS-Applied-Deep-Learning/releases/tag/v1.0.0-rc.1) and put them into the `transfer` folder.

## Flask Web App

### Local

The web app can be started locally with `python transfer/server.py`, server will run on [http://localhost:8080](http://localhost:8080).  
Docker is used for packaging.

## Docker

First a base image has to be built. It holds only the python dependencies. The second image, which depends on the base image, packages the transfered hrnet and resnet models within a flask web app, ready to be served on port 8080. The seperation into two images made the docker development round trips much shorter.

```bash
docker build -f docker-pytorch-base/Dockerfile -t adl-pytorch-base ./docker-pytorch-base/
docker build -t adl-heritage .
docker run -it --rm -p 8080:8080 adl-heritage
```

goto [http://localhost:8080](http://localhost:8080)

## Time tracking

### Planned Tasks and Times

- setup of current cuda environment @home gtx-1080ti (4h)
- check if there is a more current deep neural network trained on images which can be used for transfer learning, if there isn't a usable one use then use one from the paper, e.g. inception v3 (4h)
- get inspiration from [A Survey on Deep Transfer Learning](https://link.springer.com/chapter/10.1007/978-3-030-01424-7_27) and/or [Hands-On Transfer Learning with Python](https://proquest.tech.safaribooksonline.de/9781788831307) and/or [fast.ai](https://www.fast.ai) and/or [tensorflow](https://www.tensorflow.org/hub/tutorials/image_retraining) AND [HRNet]( https://github.com/HRNet/HRNet-Image-Classification)(8h)
- network architecture first draft including F1 score results (8h)
- train, evaluate and fine-tune the network (16h)
- web application for uploading and classifying an image with the latest model in a docker image (4h)
- final report (4h)
- TODO Exercise 3: presentation slides (4h)

### Real Tasks and Times

- setup environment (cuda & transfer example) (0.5h)
- hrnet overview and validation of pretrained model (6h)
- dummy torch test setup (0.5h)
- adapt pytorch transfer learning tutorial for heritage dataset (12h)
- include hrnet model into transfer learning for heritage dataset with F1-Score (8h)
- run experiments on all models and evaluate/document results (2h)
- refactor code to indidual files (1.5h)
- package README.md (0.5h)
- prediction of a single image (4.5h)
- dockerfile and flask (5.5h)
- final report (2.5h)

=> Total Time: 43.5h

## Learnings

### HRNet

In order to test that I can correctly use the hrnet model, I decided to validate the model on the imagenet test data.  
Here are some take aways from the process:

#### Setup of Python Dependencies

It was not necessary to have all dependencies installed for my use case. I lost a lot of time with setting up Shapely which was not needed in the end.  
It turned out that yacs and matplotlib sufficed. The hassle with shapely was not necessary at all.  
The pillow problem was not existing after clean installation of pytorch in a new python environment.

#### Dataset Loading

- if you use harddisk you should make sure that your process is the only reading from it, no other process should access the harddisk

- have multiple workers, e.g. 4 instead of only 1 (because 1 will only trash with high hd load)

- higher batch size -> more load on gpu -> more time for i/o -> optimal gpu utilization setting allows using slower harddisk (no need for ssd in this case)

best results:

```bash
ssd, workers 8, batch size 256, ~ 6.4 GB GPU Memory: 2:45 min
hd,  workers 4, batch size 128: ~ 3.0 GB GPU Memory 3:05 min
```

Main take away: I sadly confused number of workers with number of gpus and learned the hard way that you should always have more than one worker!

Second main take away: batch size optimization is important to fully utilize gpu. But if the batch size is too big, ther will be runtime memory error, latest at the second epoch. (depending on the use case disabling cudnn might help, but slow down training time, see [pytorch issue](https://github.com/pytorch/pytorch/issues/27588))

#### Project Structure

The python files in the hrnet library are loaded at runtime (e.g. `_init_paths.py`) which breaks linting and code navigation :(.

#### Cuda

Error messages like `CUDA error: device-side assert triggered` showed me, that it always important to know if the data resides on cpu or gpu and where it belongs to. This was especially necessary when applying custom scikit learn metrics (f1 score) which operate on numpy arrays on cpu.

#### Debugging

Thankfully debugging works quite well with Visual Studio Code. It was very helpful to inspect variables and to use the debug console to try out any code snippets. My debug configurations are provided in `.vscode/launch.json`.

#### Loading Models

The method of loading a model for transfer learning is to load the state from a dictionary and instantiate the correct model class.  
The saving of the transfered models worked fine, but when loading there was an exception where the weights did not match to model class anymore. I could not find the reason for the exception so I decided to transfer learn the models again and to save the models completely, in order to load the models without needing the model class anymore. This sufficed for prediction mode.

#### Testing

I tried torchtest (`pip install torchtest`), and it is surely a good way to test a neural network from scratch with unit tests.  
But the pretrained models were already tested and thus the transfered models did also not need unit test anymore because the transfer learning and tests worked positive.

## Possible Future Improvements

### Decrease size of networks

The models are very big, the hrnet model is almost 500MB in size. Maybe it is possible to prune the network.

### Only train the last layer

The transferred models were trained by updating all the weights of the pretrained model. It would be interesting to see how the scores look like if only the last layer is trained.

### Improve Web App Upload

The webapp sometimes blocks image uploads. It should allow any image to be uploaded.