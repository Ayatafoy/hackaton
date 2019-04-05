from __future__ import print_function, division
import numpy as np
import cv2
import random
import torch


class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
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

        return {'image': final_img,
                  'car_brand': sample['car_brand'],
                  'car_model': sample['car_model'],
                  'body': sample['body'],
                  'color': sample['color'],
                  'engine_type': sample['engine_type'],
                  'transmission': sample['transmission'],
                  'rudder': sample['rudder'],
                  'price': sample['price'],
                  'year': sample['year'],
                  'engine_volume': sample['engine_volume'],
                  'engine_power': sample['engine_power'],
                  }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch.tensor(np.array(image, dtype=np.long)),
                'car_brand': torch.tensor(np.array(sample['car_brand'], dtype=np.long)),
                'car_model': torch.tensor(np.array(sample['car_model'], dtype=np.long)),
                'body': torch.tensor(np.array(sample['body'], dtype=np.long)),
                'color': torch.tensor(np.array(sample['color'], dtype=np.long)),
                'engine_type': torch.tensor(np.array(sample['engine_type'], dtype=np.long)),
                'transmission': torch.tensor(np.array(sample['transmission'], dtype=np.long)),
                'rudder': torch.tensor(np.array(sample['rudder'], dtype=np.long)),
                'price': torch.tensor(np.array(sample['price'], dtype=np.long)),
                'year': torch.tensor(np.array(sample['year'], dtype=np.long)),
                'engine_volume': torch.tensor(np.array(sample['engine_volume'], dtype=np.float)),
                'engine_power': torch.tensor(np.array(sample['engine_power'], dtype=np.long)),
                }