import os
import pandas as pd
import nltk
import nltk.data
import re

nltk.download('punkt')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


class DataSetPreprocessor:
    def __init__(self):
        self.features = None
        self.label_map = None

    def get_features(self, base_dir):
        if self.features is None:
            products = os.listdir(base_dir)
            if '.DS_Store' in products:
                products.remove('.DS_Store')
            features = []
            self.label_map = {}
            for i, product in enumerate(products):
                self.label_map[i] = product
                path_to_product_dir = os.path.join(base_dir, product)
                files = os.listdir(path_to_product_dir)
                for file in files:
                    if file.endswith('.csv'):
                        path_to_csv = os.path.join(path_to_product_dir, file)
                        break
                products = pd.read_csv(path_to_csv, engine='python')
                products['path_to_img'] = products['img_id'].apply(lambda img_id: os.path.join(product, str(img_id)))
                features.append(products)
            features = pd.concat(features)
            features['car_brand_cat'] = features['car_brand'].astype('category').cat.codes.copy()
            features['car_model_cat'] = features['car_model'].astype('category').cat.codes.copy()
            features['body_cat'] = features['body'].astype('category').cat.codes.copy()
            features['color_cat'] = features['color'].astype('category').cat.codes.copy()
            features['engine_type_cat'] = features['engine_type'].astype('category').cat.codes.copy()
            features['transmission_cat'] = features['transmission'].astype('category').cat.codes.copy()
            features['rudder_cat'] = features['rudder'].astype('category').cat.codes.copy()
            features = features[['img_id', 'path_to_img', 'car_brand', 'car_model', 'body', 'color', 'engine_type',
                                 'transmission', 'rudder', 'car_brand_cat',
                                 'car_model_cat', 'body_cat', 'color_cat', 'engine_type_cat', 'transmission_cat',
                                 'rudder_cat', 'price', 'year', 'engine_volume', 'engine_power']]
            self.features = features

        return self.features


if __name__ == "__main__":
    data_dir = os.path.join("/Users/aromanov/Desktop/Cars")
    processor = DataSetPreprocessor()
    processor.get_features(data_dir)
