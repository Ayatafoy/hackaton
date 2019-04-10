import os
import numpy as np
import torch
from torch import optim
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn import model_selection
import torch.utils.data as data
import time
from DataLoader.car_data_set import CarDataSet
from DataLoader.data_set_preprocessor import DataSetPreprocessor
from Model.focal_loss import FocalLoss
from tqdm import tqdm
from Model.se_res_net import SeResNet
from Utils import models_manager


train_log = open(os.path.join(os.getcwd(), 'Train log'), "w")

def accuracy_target(true, predicted, target):
    true_samples = 0
    all_samples = 0
    for i, t in enumerate(true):
        if t == target:
            all_samples += 1
            p = predicted[i]
            if t == p:
                true_samples += 1
    if all_samples == 0:
        return -1

    return true_samples / all_samples


def evaluate_model(model,
                   classidier,
                   scheduler,
                   trainloader,
                   loader,
                   criterion,
                   device,
                   true_train,
                   predict_train,
                   running_loss,
                   epoch,
                   num_epochs,
                   epoch_start_time,
                   message):
    print(message)
    true_test = []
    predict_test = []
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            description_tokens = batch['description_tokens'].to(device)
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            classifier_logits = classidier.forward(image, description_tokens)
            logits = model.forward(classifier_logits, image, description_tokens)
            loss = criterion(logits, label)
            test_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            true_test += list(label.detach().cpu().numpy())
            predict_test += list(np.argmax(logits, axis=1))

    accuracy_kids = accuracy_target(true_test, predict_test, 0)
    accuracy_other = accuracy_target(true_test, predict_test, 1)
    accuracy = accuracy_score(true_test, predict_test)

    roc_auc = roc_auc_score(true_test, predict_test)

    num_classes = len(classes)
    matrix = confusion_matrix(true_train, predict_train, labels=range(num_classes))
    print("Train confusion matrix:")
    print(str(matrix))
    matrix = confusion_matrix(true_test, predict_test, labels=range(num_classes))
    print("Test confusion matrix:")
    print(str(matrix))
    scheduler.step(roc_auc)
    time_elapsed = time.time() - epoch_start_time
    metrics_dict = {'Epoch': str(epoch+1) + '/' + str(num_epochs),
                    'Train loss': '{:.3f}'.format(running_loss/len(trainloader)),
                    'Test loss': '{:.3f}'.format(test_loss/len(loader)),
                    'Test accuracy': '{:.3f}'.format(accuracy),
                    'Test accuracy_kids': '{:.3f}'.format(accuracy_kids),
                    'Test accuracy_other': '{:.3f}'.format(accuracy_other),
                    'Test roc_auc': '{:.3f}'.format(roc_auc),
                    'training complete in': '{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
                    }
    for metric in metrics_dict.items():
        print(metric)
    return metrics_dict


def get_batch_logits(logits, classes):
    car_brand_position = classes['car_brand']
    car_model_position = car_brand_position + classes['car_model']
    body_position = car_model_position + classes['body']
    color_position = body_position + classes['color']
    engine_type_position = color_position + classes['engine_type']
    transmission_position = engine_type_position + classes['transmission']
    rudder_position = transmission_position + classes['rudder']

    car_brand_logits = logits[:, 0:car_brand_position]
    car_model_logits = logits[:, car_brand_position:car_model_position]
    body_logits = logits[:, car_model_position: body_position]
    color_logits = logits[:, body_position:color_position]
    engine_type_logits = logits[:, color_position:engine_type_position]
    transmission_logits = logits[:, engine_type_position:transmission_position]
    rudder_logits = logits[:, transmission_position:rudder_position]
    continous_logits = logits[:, rudder_position:]

    return car_brand_logits, car_model_logits, body_logits, color_logits, engine_type_logits, transmission_logits,\
           rudder_logits, continous_logits


def criterion(car_brand_logits,
              car_model_logits,
              body_logits,
              color_logits,
              engine_type_logits,
              transmission_logits,
              rudder_logits,
              continous_logits,
              batch,
              device):
    car_brand = batch['car_brand'].to(device)
    car_model = batch['car_model'].to(device)
    body = batch['body'].to(device)
    color = batch['color'].to(device)
    engine_type = batch['engine_type'].to(device)
    transmission = batch['transmission'].to(device)
    rudder = batch['rudder'].to(device)

    price = batch['price'].to(device)
    year = batch['year'].to(device)
    engine_volume = batch['engine_volume'].to(device)
    engine_power = batch['engine_power'].to(device)

    car_brand_loss = FocalLoss(5)(car_brand_logits, car_brand)
    car_model_loss = FocalLoss(5)(car_model_logits, car_model)
    body_loss = FocalLoss(5)(body_logits, body)
    color_loss = FocalLoss(5)(color_logits, color)
    engine_type_loss = FocalLoss(5)(engine_type_logits, engine_type)
    transmission_loss = FocalLoss(5)(transmission_logits, transmission)
    rudder_loss = FocalLoss(5)(rudder_logits, rudder)

    price_loss = criterion_mse(continous_logits[:, 0], price)
    year_loss = criterion_mse(continous_logits[:, 1], year)
    engine_volume_loss = criterion_mse(continous_logits[:, 2], engine_volume)
    engine_power_loss = criterion_mse(continous_logits[:, 3], engine_power)

    train_log.write('car_brand_loss: %s' % car_brand_loss.item() + '\n')
    train_log.write('car_model_loss: %s' % car_model_loss.item() + '\n')
    train_log.write('body_loss: %s' % body_loss.item() + '\n')
    train_log.write('color_loss: %s' % color_loss.item() + '\n')
    train_log.write('engine_type_loss: %s' % engine_type_loss.item() + '\n')
    train_log.write('transmission_loss: %s' % transmission_loss.item() + '\n')
    train_log.write('rudder_loss: %s' % rudder_loss.item() + '\n')

    train_log.write('price_loss: %s' % price_loss.item() + '\n')
    train_log.write('year_loss: %s' % year_loss.item() + '\n')
    train_log.write('engine_volume_loss: %s' % engine_volume_loss.item() + '\n')
    train_log.write('engine_power_loss: %s' % engine_power_loss.item() + '\n')

    total_loss = car_brand_loss + car_model_loss + body_loss + color_loss + engine_type_loss + transmission_loss + \
                 rudder_loss + price_loss + year_loss + engine_volume_loss + engine_power_loss

    return total_loss


def train(model, classes, optimizer,
          scheduler, model_dir, trainloader, testloader, validation_loader, device,
          num_epochs=5, n_epochs_stop=10):
    model.to(device)
    epochs_no_improve = 0
    best_acc = -np.inf
    path_for_best_model = ''
    for epoch in tqdm(range(num_epochs)):
        epoch_start_time = time.time()
        print("Epoch %s started..." % epoch)
        true_car_brand = []
        predict_car_brand = []
        true_car_model = []
        predict_car_model = []
        true_body = []
        predict_body = []
        true_color = []
        predict_color = []
        true_engine_type = []
        predict_engine_type = []
        true_transmission = []
        predict_transmission = []
        true_rudder = []
        predict_rudder = []
        running_loss = 0
        for batch in tqdm(trainloader):
            image = batch['image'].to(device)
            car_brand = batch['car_brand'].to(device)
            car_model = batch['car_model'].to(device)
            body = batch['body'].to(device)
            color = batch['color'].to(device)
            engine_type = batch['engine_type'].to(device)
            transmission = batch['transmission'].to(device)
            rudder = batch['rudder'].to(device)
            optimizer.zero_grad()
            logits = model.forward(image)
            car_brand_logits, car_model_logits, body_logits, color_logits, engine_type_logits, transmission_logits,\
            rudder_logits, continous_logits = get_batch_logits(logits, classes)
            loss = criterion(car_brand_logits, car_model_logits, body_logits, color_logits, engine_type_logits,
                             transmission_logits, rudder_logits, continous_logits, batch, device)

            true_car_brand += list(car_brand.detach().cpu().numpy())
            predict_car_brand += list(np.argmax(car_brand_logits.detach().cpu().numpy(), axis=1))

            true_car_model += list(car_model.detach().cpu().numpy())
            predict_car_model += list(np.argmax(car_model_logits.detach().cpu().numpy(), axis=1))

            true_body += list(body.detach().cpu().numpy())
            predict_body += list(np.argmax(body_logits.detach().cpu().numpy(), axis=1))

            true_color += list(color.detach().cpu().numpy())
            predict_color += list(np.argmax(color_logits.detach().cpu().numpy(), axis=1))

            true_engine_type += list(engine_type.detach().cpu().numpy())
            predict_engine_type += list(np.argmax(engine_type_logits.detach().cpu().numpy(), axis=1))

            true_transmission += list(transmission.detach().cpu().numpy())
            predict_transmission += list(np.argmax(transmission_logits.detach().cpu().numpy(), axis=1))

            true_rudder += list(rudder.detach().cpu().numpy())
            predict_rudder += list(np.argmax(rudder_logits.detach().cpu().numpy(), axis=1))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Loss: {}".format(running_loss / len(trainloader)))
        accuracy_car_brand = accuracy_score(true_car_brand, predict_car_brand)
        print("Accuracy car brand: {}".format(accuracy_car_brand))
        accuracy_car_model = accuracy_score(true_car_model, predict_car_model)
        print("Accuracy car model: {}".format(accuracy_car_model))
        accuracy_body = accuracy_score(true_body, predict_body)
        print("Accuracy body: {}".format(accuracy_body))
        accuracy_color = accuracy_score(true_color, predict_color)
        print("Accuracy color: {}".format(accuracy_color))
        accuracy_engine_type = accuracy_score(true_engine_type, predict_engine_type)
        print("Accuracy engine type: {}".format(accuracy_engine_type))
        accuracy_transmission = accuracy_score(true_transmission, predict_transmission)
        print("Accuracy transmission: {}".format(accuracy_transmission))
        accuracy_rudder = roc_auc_score(true_rudder, predict_rudder)
        print("Roc auc score rudder: {}".format(accuracy_rudder))
        scheduler.step(accuracy_car_model)
        if accuracy_car_model > best_acc:
            epochs_no_improve = 0
            best_acc = accuracy_car_model
            models_manager.save_checkpoint({'epoch': epoch + 1,
                                           'state_dict': model.state_dict(),
                                           'optim_dict': optimizer.state_dict()},
                                           checkpoint=model_dir)

            path_for_best_model = os.path.join(model_dir, 'AvitoAdGenerator')
            path_for_best_model = path_for_best_model + "_accuracy_{}.pth".format(accuracy_car_model)
            torch.save(model.state_dict(), path_for_best_model)
            print("Best model \'AvitoAdGenerator.pth\' saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                break
        print("Epoch %s: takes %s seconds" % (epoch, (time.time() - epoch_start_time)))
        path_for_best_model = os.path.join(model_dir, 'AvitoAdGenerator.pth')
        torch.save(model.state_dict(), path_for_best_model)
    return path_for_best_model


def run_train(trainloader,
              testloader,
              validation_loader,
              model_dir,
              classes,
              num_classes,
              path_to_pretrained_resnet=None,
              cuda_device=1,
              optimizer_lr=0.01,
              num_epohs_top=5,
              num_epohs_total=50):
    if cuda_device is not None:
        assert cuda_device == 0 or cuda_device == 1

    device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")
    resnet = SeResNet(num_classes)
    if path_to_pretrained_resnet is not None:
        resnet_state_dict = torch.load(path_to_pretrained_resnet)
        resnet.load_state_dict(resnet_state_dict)
    resnet.set_gr(False)

    optimizer = optim.Adam(resnet.parameters(), lr=optimizer_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    train(model=resnet,
          classes=classes,
          optimizer=optimizer,
          scheduler=scheduler,
          model_dir=model_dir,
          trainloader=trainloader,
          testloader=testloader,
          validation_loader=validation_loader,
          device=device,
          num_epochs=num_epohs_top,
          n_epochs_stop=num_epohs_total)

    resnet.set_gr(True)

    path_for_best_model = train(model=resnet,
                                classes=classes,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                model_dir=model_dir,
                                trainloader=trainloader,
                                testloader=testloader,
                                validation_loader=validation_loader,
                                device=device,
                                num_epochs=num_epohs_total,
                                n_epochs_stop=num_epohs_total)
    return path_for_best_model


params = {
    'num_epohs_top': 0,
    'num_epohs_total': 200,
    'optimizer_lr': 0.00001,
    'cuda_device': 0,
}

path_to_train_data = '/Users/aromanov/Desktop/Scrapped data/Cars'
processor = DataSetPreprocessor()
features = processor.get_features(path_to_train_data)

print(features['car_brand'].value_counts())
print(features['car_model'].value_counts())
print(features['body'].value_counts())
print(features['color'].value_counts())
print(features['engine_type'].value_counts())
print(features['transmission'].value_counts())
print(features['rudder'].value_counts())

car_brand_cat = len(features['car_brand_cat'].unique())
car_model_cat = len(features['car_model_cat'].unique())
body_cat = len(features['body_cat'].unique())
color_cat = len(features['color_cat'].unique())
engine_type_cat = len(features['engine_type_cat'].unique())
transmission_cat = len(features['transmission_cat'].unique())
rudder_cat = len(features['rudder_cat'].unique())

classes ={
    'car_brand': car_brand_cat,
    'car_model': car_model_cat,
    'body': body_cat,
    'color': color_cat,
    'engine_type': engine_type_cat,
    'transmission': transmission_cat,
    'rudder': rudder_cat
}

num_classes = car_brand_cat + car_model_cat + body_cat + color_cat + engine_type_cat + transmission_cat + rudder_cat + 4
print("Num_classes: {}".format(num_classes))
# print('Num of car_brand_cat: {}'.format(car_brand_cat))
# print('Num of car_model_cat: {}'.format(car_model_cat))
# print('Num of body_cat: {}'.format(body_cat))
# print('Num of color_cat: {}'.format(color_cat))
# print('Num of engine_type_cat: {}'.format(engine_type_cat))
# print('Num of transmission_cat: {}'.format(transmission_cat))
# print('Num of rudder_cat: {}'.format(rudder_cat))
criterion_mse = torch.nn.MSELoss()

train_features, validation_features = model_selection.train_test_split(features, random_state=0)
model_dir = os.path.join(os.getcwd(), 'models')
path_to_pretrained_resnet = os.path.join(model_dir, 'AvitoAdGenerator_accuracy_0.8524310491896503.pth')
train_features, test_features = model_selection.train_test_split(train_features, test_size=0.1)
train_data = CarDataSet(train_features, path_to_train_data)
test_data = CarDataSet(test_features, path_to_train_data)
validation_data = CarDataSet(validation_features, path_to_train_data)
train_loader = data.DataLoader(train_data, shuffle=True, batch_size=2)
test_loader = data.DataLoader(test_data, shuffle=False, batch_size=32)
validation_loader = data.DataLoader(validation_data, shuffle=False, batch_size=32)
run_train(train_loader,
          test_loader,
          validation_loader,
          model_dir,
          classes,
          num_classes,
          None,
          params['cuda_device'],
          params['optimizer_lr'],
          params['num_epohs_top'],
          params['num_epohs_total'])