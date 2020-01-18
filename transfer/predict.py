import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os
from PIL import Image
from torch.autograd import Variable

from model_trainer import train_model
from plots import visualize_data
from config_builder import from_args
from model_provider import provide_model
import models

# https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
def predict_image(image, device):
    test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

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

    class_names = ['altar','apse','bell tower','column','dome(inner)','dome(outer)','flying_buttress','gargoyle','stained_glass','vault']
    # device = torch.device("cpu")
    # model = provide_model(model_name, len(class_names), device).to(device)
    # model.load_state_dict(torch.load(os.path.join('./hrnet_final_state_largest_94_epochs10.pth.tar')))
    # model.eval()

    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    model = torch.load(os.path.join('./' + model_name + '_final_model.pt')).to(device)
    model.eval()

    image_path = '../../data/heritage/test/altar/0a020a2f-e72a-4663-ae34-4d236e7c5ea2.jpg'
    output = model(data_transforms['test'](Image.open(image_path)).unsqueeze_(0).to(device))
    print(f'{class_names[int(torch.max(output.data.cpu(), 1)[1].numpy())]}')
    image_path = '../../data/heritage/test/altar/157eb411-7954-4029-83e1-bc857f97c5dd.jpg'
    output = model(data_transforms['test'](Image.open(image_path)).unsqueeze_(0).to(device))
    print(f'{class_names[int(torch.max(output.data.cpu(), 1)[1].numpy())]}')
    image_path = '../../data/heritage/test/dome(inner)/36412379281_0b85042e9c_n.jpg'
    output = model(data_transforms['test'](Image.open(image_path)).unsqueeze_(0).to(device))
    print(f'{class_names[int(torch.max(output.data.cpu(), 1)[1].numpy())]}')

if __name__ == '__main__':
    main()
