# Time tracking

## Planned Times

Work Breakdown

- setup of current cuda environment @home gtx-1080ti (4h)
- check if there is a more current deep neural network trained on images which can be used for transfer learning, if there isn't a usable one use then use one from the paper, e.g. inception v3 (4h)
- get inspiration from [A Survey on Deep Transfer Learning](https://link.springer.com/chapter/10.1007/978-3-030-01424-7_27) and/or [Hands-On Transfer Learning with Python](https://proquest.tech.safaribooksonline.de/9781788831307) and/or [fast.ai](https://www.fast.ai) and/or [tensorflow](https://www.tensorflow.org/hub/tutorials/image_retraining) AND [HRNet]( https://github.com/HRNet/HRNet-Image-Classification)(8h)
- network architecture first draft including F1 score results (8h)
- train, evaluate and fine-tune the network (16h)
- web application for uploading and classifying an image with the latest model in a docker image (4h)
- final report (4h)
- presentation slides (4h)

## Real Times

Work Breakdown

- setup environment (cuda & transfer example) (0.5h)
- hrnet overview and validation of pretrained model (6h)
- torch test setup (0.5h)
- adapt pytorch transfer learning tutorial for heritage dataset (12h)
- include hrnet model into transfer learning for heritage dataset with F1-Score (8h)
- run experiments on all models and evaluate/document results (2h)
- refactor code to own files (2h)
- prediction of a single image (4.5h)
- dockerfile and flask (5.5h)