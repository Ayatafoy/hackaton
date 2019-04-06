from __future__ import print_function, division
import pandas as pd
import numpy as np
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from DataLoader.DataSetPreprocessor import DataSetPreprocessor
from sklearn import model_selection
from DataLoader.Transformations import Resize, ToTensor
import torch.utils.data as data


class CarDataSet(Dataset):

    def __init__(self, features, path_to_images, transform=transforms.Compose([Resize(224), ToTensor()])):
        self.features = features.values
        self.path_to_images = path_to_images
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        img_path = self.features[idx, 1]
        img_name = os.path.join(self.path_to_images, str(img_path) + ".jpg")
        try:
            img = Image.open(img_name)
        except FileNotFoundError:
            img_name = os.path.join(self.path_to_images, str(img_path) + ".png")
            img = Image.open(img_name)
        image = img.convert('RGB')
        car_brand = self.features[idx, 9]
        car_model = self.features[idx, 10]
        body = self.features[idx, 11]
        color = self.features[idx, 12]
        engine_type = self.features[idx, 13]
        transmission = self.features[idx, 14]
        rudder = self.features[idx, 15]
        price = self.features[idx, 16]
        year = self.features[idx, 17]
        engine_volume = self.features[idx, 18]
        engine_power = self.features[idx, 19]

        sample = {'image': image,
                  'car_brand': car_brand,
                  'car_model': car_model,
                  'body': body,
                  'color': color,
                  'engine_type': engine_type,
                  'transmission': transmission,
                  'rudder': rudder,
                  'price': price,
                  'year': year,
                  'engine_volume': engine_volume,
                  'engine_power': engine_power
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_classes(self):
        classes = set(self.features[:, 1])

        return classes


if __name__ == "__main__":
    data_dir = os.path.join("/Users/aromanov/Desktop/Cars")
    processor = DataSetPreprocessor()
    features = processor.get_features(data_dir)
    print(len(features))
    input_data_set = CarDataSet(features, data_dir)
    train_data, test_data = model_selection.train_test_split(input_data_set, test_size=0.1, random_state=0)