# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

from model_trainer import train_model
from plots import visualize_data
from config_builder import from_args
from model_provider import provide_model


def main():
    config = from_args()

    model_name = config.MODEL_FAMILY
    batch_size = config.TRANSFER_BATCH_SIZE
    num_workers = config.WORKERS
    num_epochs = config.TRANSFER_EPOCHS
    data_dir = config.TRANSFER_DATA_FOLDER
    visualize = config.VISUALIZE_CLASSES

    # reproducability (https://pytorch.org/docs/stable/notes/randomness.html)
    torch.manual_seed(73)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = provide_model(model_name, len(class_names), device).to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           image_datasets, dataloaders, device, num_epochs=num_epochs)

    final_model_state_file = os.path.join('./',
                                          model_name + '_final_state.pth.tar')
    print('saving final model state to {}'.format(final_model_state_file))
    torch.save(model_ft.state_dict(), final_model_state_file)

    if visualize:
        visualize_data(model_ft, class_names,
                       dataloaders, device, num_images=10)


if __name__ == '__main__':
    main()
