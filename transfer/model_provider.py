import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import os

import _add_hrnet_lib # needed for import of hrnet library classes
# hrnet library classes
from core.function import validate
from config import config
from config import update_config
from utils.modelsummary import get_model_summary

def load_hrnet_pretrained():
    # cudnn related setting
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

    device = torch.device("cuda:0")
    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    print(get_model_summary(model.cuda(device), dump_input.cuda(device), verbose=True))

    print('=> loading model from {}'.format(config.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(config.TEST.MODEL_FILE))

    return model

def provide_model(model_name, classes_count, device, full=True):
    """
    provides the pretrained model by model_name and replaces last classification layer with number of classes
    if full is provided and set to False, then the weights of the pretrained are set unmodifiable
    """
    if model_name == 'hrnet':
        hrnet_model = load_hrnet_pretrained()
        num_ftrs = hrnet_model.classifier.in_features
        hrnet_model.classifier = nn.Linear(num_ftrs, classes_count)
        model = nn.Sequential(hrnet_model).cuda(device)
    elif model_name == 'resnet18':
        resnet18 = torchvision.models.resnet18(pretrained=True)
        num_ftrs = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_ftrs, classes_count)
        model = resnet18
    elif model_name == 'resnet152':
        resnet152 = torchvision.models.resnet152(pretrained=True)
        num_ftrs = resnet152.fc.in_features
        resnet152.fc = nn.Linear(num_ftrs, classes_count)
        model = resnet152

    dump_input = torch.rand((1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]))
    print(get_model_summary(model.cpu(), dump_input.cpu(), verbose=True))
    return model

def validate_hrnet():
    # cudnn related setting
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

    device = torch.device("cuda:0")
    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    print(get_model_summary(model.cuda(device), dump_input.cuda(device), verbose=True))

    if config.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join("./",
                                        'final_state.pth.tar')
        print('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # model = torch.nn.Sequential(model).cuda()
    model = torch.nn.Sequential(model).cuda(device)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda(device)

    # Data loading code
    valdir = os.path.join(config.DATASET.ROOT,
                          config.DATASET.TEST_SET)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ]))

    print(f'number of images: {len(dataset.imgs)}')

    valid_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(config, valid_loader, model, criterion, './log',
             '/output', None)
