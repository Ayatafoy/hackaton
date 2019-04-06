import pandas as pd
import os
import requests
from PIL import Image
from base64 import b64decode
from io import BytesIO
import re


def load_picture(path_to_pict, pict_name, path_to_load):
    if path_to_pict.startswith('data:image/webp'):
        img_data = path_to_pict[len("data:image/webp;base64,"):]
        im = Image.open(BytesIO(b64decode(img_data)))
    else:
        im = Image.open(requests.get(path_to_pict, stream=True).raw)
    path_to_pict_load = os.path.join(path_to_load, str(pict_name) + '.jpg')
    im.save(path_to_pict_load)


base_dir = "/Users/aromanov/Desktop/Cars new"
cars = os.listdir(base_dir)
if '.DS_Store' in cars:
    cars.remove('.DS_Store')
cars = [car[0:-4] for car in cars]
for car_name in cars:
    path_to_product_data = os.path.join("/Users/aromanov/Desktop/Cars new", car_name)
    if os.path.exists(path_to_product_data):
        for the_file in os.listdir(path_to_product_data):
            file_path = os.path.join(path_to_product_data, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    else:
        os.mkdir(path_to_product_data)

    path_to_item_features = "/Users/aromanov/Desktop/Cars new/{}.csv".format(car_name)
    products = pd.read_csv(path_to_item_features, engine='python')
    products = products.fillna('')
    cars = pd.DataFrame(columns=['img_id', 'car_brand', 'car_model', 'price', 'year', 'body', 'color',
                                 'engine_volume', 'engine_power', 'engine_type', 'transmission', 'rudder'])
    for i, product in enumerate(products.itertuples()):
        try:
            img_id = product.web_scraper_order[:product.web_scraper_order.find('-')]
            img_src = 'http:' + product.image_src
            href_parts = product.car_href.split('/')
            engine_parts = product.engine.split('/')
            engine_volume = re.findall("\d+\.\d+", engine_parts[0])[0]
            engine_power = re.findall("\d+", engine_parts[1])[0]
            engine_type = engine_parts[2]
            car_brand = href_parts[6]
            car_model = href_parts[7]
            price_parts = re.findall("\d+", product.price.replace(' ', ''))
            price = ''
            for p in price_parts:
                price += p
            year = product.year
            body = product.body
            load_picture(img_src, img_id, path_to_product_data)

            cars = cars.append({'img_id': img_id, 'car_brand': car_brand, 'car_model': car_model, 'price': int(price),
                                   'year': int(product.year), 'body': product.body, 'color': product.color,
                                   'engine_volume': float(engine_volume), 'engine_power': int(engine_power),
                                   'engine_type': engine_type,
                                   'transmission': product.transmission,
                                   'rudder': product.rudder}, ignore_index=True)
            print("Counter: {}".format(i))
        except Exception as e:
            print("Exception! {}".format(e))
            continue

    cars.to_csv(os.path.join(path_to_product_data, car_name + "_attributes.csv"), index=False)


