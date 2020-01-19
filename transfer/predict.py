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


class TransferModel:
    

    def __init__(self, folder, model_file):
        self.class_names = ['altar','apse','bell tower','column','dome(inner)','dome(outer)','flying_buttress','gargoyle','stained_glass','vault']
        self.model_name = 'resnet152'
        self.device = torch.device("cpu")
        self.folder = folder
        self.model = torch.load(os.path.join(self.folder + model_file), map_location=self.device)
        self.model.eval()


    # https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
    def predict_image(self, image, device):
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

    def from_output(self, output):
        class_name = self.class_names[int(torch.max(output.data.cpu(), 1)[1].numpy())]
        class_probability = torch.max(torch.nn.functional.softmax(output.data.cpu(), dim=1)).item()
        print(f'predicting {class_name} with a probability of {round(class_probability,4)*100}%')
        return (class_name, class_probability)

    def predict(self, image = None):
        # Just normalization for test
        data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # image_path = '../../data/heritage/test/dome(inner)/36412379281_0b85042e9c_n.jpg'
        # output = model(data_transforms['test'](Image.open(image_path)).unsqueeze_(0).to(device))
        # class_name, class_probability = from_output(output)

        if image is None:
            image_path = self.folder + 'St._Dionysius,_Rheine_-_Josef-Altar.jpg'
            output = self.model(data_transforms['test'](Image.open(image_path)).unsqueeze_(0).to(self.device))
            return self.from_output(output)
        else:
            output = self.model(data_transforms['test'](image).unsqueeze_(0).to(self.device))
            return self.from_output(output)


def main():
    config = from_args()

    model = TransferModel('./', 'resnet152_final_model.pt')
    hrnet_model = TransferModel('./', 'hrnet_final_model.pt')

    model.predict()
    hrnet_model.predict()

if __name__ == '__main__':
    main()
