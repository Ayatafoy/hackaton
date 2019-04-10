import os
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler


class DataSetPreprocessor:
    def __init__(self):
        self.features = None
        self.label_map = None
        self.scaler = MaxAbsScaler()

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

            features[['price', 'year', 'engine_volume', 'engine_power']] = self.scaler.fit_transform(features[['price', 'year', 'engine_volume', 'engine_power']].values)
            self.features = features

        return self.features


if __name__ == "__main__":
    data_dir = os.path.join("/Users/aromanov/Desktop/Cars")
    processor = DataSetPreprocessor()
    features = processor.get_features(data_dir)
    car_brand_mapping = {}
    car_model_mapping = {}
    body_mapping = {}
    color_mapping = {}
    engine_type_mapping = {}
    transmission_mapping = {}
    rudder_mapping = {}
    for row in features.itertuples():
        car_brand_mapping[row.car_brand_cat] = row.car_brand
        car_model_mapping[row.car_model_cat] = row.car_model
        body_mapping[row.body_cat] = row.body
        color_mapping[row.color_cat] = row.color
        engine_type_mapping[row.engine_type_cat] = row.engine_type
        transmission_mapping[row.transmission_cat] = row.transmission
        rudder_mapping[row.rudder_cat] = row.rudder
    x = 0
