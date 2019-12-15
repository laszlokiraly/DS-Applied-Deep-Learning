# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse

import _add_hrnet_lib # needed for import of hrnet library classes
# hrnet library classes
import models
from core.function import validate
from config import config
from config import update_config
from utils.modelsummary import get_model_summary

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    # hrnet arguments
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    # custom arguments
    parser.add_argument('--modelFamily',
                        help='modelFamily, default is hrnet. other otions are resnet18 and resnet152',
                        type=str,
                        default='hrnet')
    parser.add_argument('--transferEpochs',
                        help='transferEpochs, number of epcochs used for transfer learning',
                        type=int,
                        default=5)
    parser.add_argument('--transferBatchSize',
                        help='transferBatchSize, batch size used for transfer learning',
                        type=int,
                        default=5)
    parser.add_argument('--transferDataFolder',
                        help='transferDataFolder, default is hrnet. other otions are resnet18 and resnet152',
                        type=str,
                        default='../../data/heritage')
    args = parser.parse_args()

    # hrnet config util
    update_config(config, args)

    # custom arguments
    config.defrost()
    if args.modelFamily:
        config.MODEL_FAMILY = args.modelFamily
    if args.transferEpochs:
        config.TRANSFER_EPOCHS = args.transferEpochs
    if args.transferBatchSize:
        config.TRANSFER_BATCH_SIZE = args.transferBatchSize
    if args.transferDataFolder:
        config.TRANSFER_DATA_FOLDER = args.transferDataFolder
    config.freeze()

    return args

def load_hrnet_pretrained(args):
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


def validate_hrnet(args):
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

def main():
    args = parse_args()

    # validate_hrnet(args)

    model_name = config.MODEL_FAMILY
    # resnet18: 256
    # resnet152: 32 worked, 48 did not
    # hrnet: small v1 128, largest: 20
    batch_size = config.TRANSFER_BATCH_SIZE 
    num_workers = config.WORKERS
    # resnets are done after 5 epochs
    num_epochs = config.TRANSFER_EPOCHS

    data_dir = config.TRANSFER_DATA_FOLDER

    # reproducability (https://pytorch.org/docs/stable/notes/randomness.html)
    torch.manual_seed(73)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                pin_memory=True, shuffle=True, num_workers=num_workers)
                for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


    visualize = False
    if visualize:
        # interactive
        plt.ion()
        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[x] for x in classes])


    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['test']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    def provide_model(model_name, classes_count, args, full=True):
        """
        provides the pretrained model by model_name and replaces last classification layer with number of classes
        if full is provided and set to False, then the weights of the pretrained are set unmodifiable
        """
        if model_name == 'hrnet':
            hrnet_model = load_hrnet_pretrained(args)
            num_ftrs = hrnet_model.classifier.in_features
            hrnet_model.classifier = nn.Linear(num_ftrs, classes_count)
            model = nn.Sequential(hrnet_model).cuda(device)
            return model
        elif model_name == 'resnet18':
            resnet18 = torchvision.models.resnet18(pretrained=True)
            num_ftrs = resnet18.fc.in_features
            resnet18.fc = nn.Linear(num_ftrs, classes_count)
            return resnet18
        elif model_name == 'resnet152':
            resnet152 = torchvision.models.resnet152(pretrained=True)
            num_ftrs = resnet152.fc.in_features
            resnet152.fc = nn.Linear(num_ftrs, classes_count)
            return resnet152

    model_ft = provide_model(model_name, len(class_names), args).to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

    final_model_state_file = os.path.join('./',
                                            model_name + '_final_state.pth.tar')
    print('saving final model state to {}'.format(final_model_state_file))
    torch.save(model_ft.state_dict(), final_model_state_file)

    if visualize:
        visualize_model(model_ft, num_images=20)
        plt.ioff()
        plt.show()



if __name__ == '__main__':
    main()
