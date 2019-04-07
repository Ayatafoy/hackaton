import os
from PIL import Image
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from Model.SeResNet import SeResNet
import numpy as np
import math
import cv2
import random
import torch
import pickle
from flask import Flask, render_template, request, jsonify, send_file


class AdGenerator:
    def __init__(self):
        path_to_pretrained_resnet = '/Users/aromanov/PycharmProjects/Ottepel/SavedModels/AvitoAdGenerator_accuracy_0.9337503554165482.pth'
        self.resnet = SeResNet(249)
        self.resnet_state_dict = torch.load(path_to_pretrained_resnet, map_location='cpu')
        self.resnet.load_state_dict(self.resnet_state_dict)
        with open('/Users/aromanov/PycharmProjects/Ottepel/SavedModels/Scaller', 'rb') as f:
            self.scaler = pickle.load(f)
        with open('/Users/aromanov/PycharmProjects/Ottepel/SavedModels/classes', 'rb') as f:
            self.classes = pickle.load(f)
        with open('/Users/aromanov/PycharmProjects/Ottepel/SavedModels/features', 'rb') as f:
            self.features = pickle.load(f)
        self.features = pd.DataFrame(self.features, index=range(len(self.features)), columns=['img_id', 'path_to_img', 'car_brand', 'car_model', 'body', 'color', 'engine_type',
                                 'transmission', 'rudder', 'car_brand_cat',
                                 'car_model_cat', 'body_cat', 'color_cat', 'engine_type_cat', 'transmission_cat',
                                 'rudder_cat', 'price', 'year', 'engine_volume', 'engine_power'])
        self.car_brand_mapping = {}
        self.car_model_mapping = {}
        self.body_mapping = {}
        self.color_mapping = {}
        self.engine_type_mapping = {}
        self.transmission_mapping = {}
        self.rudder_mapping = {}
        for row in self.features.itertuples():
            self.car_brand_mapping[row.car_brand_cat] = row.car_brand
            self.car_model_mapping[row.car_model_cat] = row.car_model
            self.body_mapping[row.body_cat] = row.body
            self.color_mapping[row.color_cat] = row.color
            self.engine_type_mapping[row.engine_type_cat] = row.engine_type
            self.transmission_mapping[row.transmission_cat] = row.transmission
            self.rudder_mapping[row.rudder_cat] = row.rudder
        self.output_size = 224

    def parse_logits(self, logits, classes, file_id):
        car_brand_position = classes['car_brand']
        car_model_position = car_brand_position + classes['car_model']
        body_position = car_model_position + classes['body']
        color_position = body_position + classes['color']
        engine_type_position = color_position + classes['engine_type']
        transmission_position = engine_type_position + classes['transmission']
        rudder_position = transmission_position + classes['rudder']

        car_brand_logits = logits[0:car_brand_position]
        car_model_logits = logits[car_brand_position:car_model_position]
        body_logits = logits[car_model_position: body_position]
        color_logits = logits[body_position:color_position]
        engine_type_logits = logits[color_position:engine_type_position]
        transmission_logits = logits[engine_type_position:transmission_position]
        rudder_logits = logits[transmission_position:rudder_position]
        continous_logits = logits[rudder_position:]

        car_brand_id = np.argmax(car_brand_logits.detach().cpu().numpy(), axis=-1)
        car_model_id = np.argmax(car_model_logits.detach().cpu().numpy(), axis=-1)
        body_id = np.argmax(body_logits.detach().cpu().numpy(), axis=-1)
        color_id = np.argmax(color_logits.detach().cpu().numpy(), axis=-1)
        engine_type_id = np.argmax(engine_type_logits.detach().cpu().numpy(), axis=-1)
        transmission_id = np.argmax(transmission_logits.detach().cpu().numpy(), axis=-1)
        rudder_id = np.argmax(rudder_logits.detach().cpu().numpy(), axis=-1)

        # print(continous_logits.shape)
        # print(continous_logits.detach().numpy().reshape(1, 4).shape)
        continous_values = continous_logits.detach().numpy().reshape(1, 4)
        continous_values = np.concatenate((continous_values, continous_values, continous_values, continous_values), axis=0)
        continous_values = self.scaler.inverse_transform(continous_values)

        ad_json = {
            "car_brand": self.car_brand_mapping[car_brand_id],
            "car_model": self.car_model_mapping[car_model_id],
            "price": math.fabs(continous_values[0][0]),
            "year": math.fabs(continous_values[0][1]),
            "body": self.body_mapping[body_id],
            "color": self.color_mapping[color_id],
            "engine_volume": math.fabs(continous_values[0][2]),
            "engine_power": math.fabs(continous_values[0][3]),
            "engine_type": self.engine_type_mapping[engine_type_id],
            "transmission": self.transmission_mapping[transmission_id],
            "rudder": self.rudder_mapping[rudder_id],
            "_id": file_id,
        }

        return ad_json

    def get_ad_json(self, path_to_image, file_id):
        img = Image.open(path_to_image)
        image = img.convert('RGB')
        image = np.asarray(image)
        scale = self.output_size / max(image.shape[:2])
        sub_img = cv2.resize(image, dsize=None, fx=scale, fy=scale)
        final_img = np.zeros((self.output_size, self.output_size, 3))
        final_img[:, :, :] = 0
        x_offset = random.randint(0, final_img.shape[0] - sub_img.shape[0])
        y_offset = random.randint(0, final_img.shape[1] - sub_img.shape[1])
        final_img[x_offset:x_offset + sub_img.shape[0], y_offset:y_offset + sub_img.shape[1]] = sub_img
        final_img = np.array(final_img, dtype=np.double)
        final_img = np.divide(final_img, 255)
        final_img = final_img.transpose((2, 0, 1))
        img_tensor = torch.tensor(np.array(final_img, dtype=np.float32))
        img_tensor = img_tensor.unsqueeze(0)
        batch = torch.cat((img_tensor, img_tensor), 0)
        logits = self.resnet(batch)

        ad_json = self.parse_logits(logits[0], self.classes, file_id)

        return ad_json


if __name__ == "__main__":
    img_name = os.path.join("/Users/aromanov/PycharmProjects/Ottepel/uploads/0x37dd7ed4e0a94ae43a384ffe97bc4054.jpg")
    generator = AdGenerator()
    logits = generator.get_ad_json(img_name, 0)

