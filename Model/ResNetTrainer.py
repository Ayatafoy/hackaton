import os
import numpy as np
import torch
from torch import optim
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn import model_selection
import torch.utils.data as data
import time
from DataLoader.CarDataSet import CarDataSet
from DataLoader.DataSetPreprocessor import DataSetPreprocessor
from Model.FocalLoss import FocalLoss
from Model.SeResNet import SeResNet
from Utils import ModelsManager


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


def criterion(logits, classes, batch, device):
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
    price_logits = logits[:, -4]
    year_logits = logits[:, -3]
    engine_volume_logits = logits[:, -2]
    engine_power_logits = logits[:, -1]

    car_brand_loss = FocalLoss(5)(car_brand_logits, car_brand)
    car_model_loss = FocalLoss(5)(car_model_logits, car_model)
    body_loss = FocalLoss(5)(body_logits, body)
    color_loss = FocalLoss(5)(color_logits, color)
    engine_type_loss = FocalLoss(5)(engine_type_logits, engine_type)
    transmission_loss = FocalLoss(5)(transmission_logits, transmission)
    rudder_loss = FocalLoss(5)(rudder_logits, rudder)
    price_loss = torch.nn.MSELoss()(price_logits, price)
    year_loss = torch.nn.MSELoss()(year_logits, year)
    engine_volume_loss = torch.nn.MSELoss()(engine_volume_logits, engine_volume)
    engine_power_loss = torch.nn.MSELoss()(engine_power_logits, engine_power)

    # total_loss = car_brand_loss + car_model_loss + body_loss + color_loss + engine_type_loss + transmission_loss + rudder_loss + price_loss + year_loss + engine_volume_loss + engine_power_loss

    return price_loss


def train(model, classes, optimizer,
          scheduler, model_dir, trainloader, testloader, validation_loader, device,
          num_epochs=5, n_epochs_stop=10):
    model.to(device)
    steps = 0
    running_loss = 0
    epochs_no_improve = 0
    best_roc_auc = -np.inf
    path_for_best_model = ''
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print("Epoch %s started..." % epoch)
        true_train = []
        predict_train = []

        for batch in trainloader:
            image = batch['image'].to(device)
            optimizer.zero_grad()
            logits = model.forward(image)
            loss = criterion(logits, classes, batch, device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # true_train += list(label.detach().cpu().numpy())
            # logits = logits.detach().cpu().numpy()
            # predict_train += list(np.argmax(logits, axis=1))
            # steps += 1
        # metrics_dict = evaluate_model(model,
        #                               classidier,
        #                               scheduler,
        #                               trainloader,
        #                               testloader,
        #                               device,
        #                               true_train,
        #                               predict_train,
        #                               running_loss,
        #                               epoch,
        #                               num_epochs,
        #                               epoch_start_time,
        #                               "Test evaluation...")
        # current_test_roc_auc = float(metrics_dict['Test roc_auc'])
        # if current_test_roc_auc > best_roc_auc:
        #     epochs_no_improve = 0
        #     best_roc_auc = current_test_roc_auc
        #     ModelsManager.save_checkpoint({'epoch': epoch + 1,
        #                                    'state_dict': model.state_dict(),
        #                                    'optim_dict': optimizer.state_dict()},
        #                                     checkpoint=model_dir)
        #
            # path_for_best_model = os.path.join(model_dir, 'Luggage_classifier/Luggage_classifier')
            # path_for_best_model = path_for_best_model + "_roc_auc_{}.pth".format(current_test_roc_auc)
            # torch.save(model.state_dict(), path_for_best_model)
        #     print("Best model \'Luggage_classifier_last.pth\' saved!")
        #     best_json_path = os.path.join(model_dir, "metrics_val_best_weights_last.json")
        #     ModelsManager.save_dict_to_json(metrics_dict, best_json_path)
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve == n_epochs_stop:
        #         print('Early stopping!')
        #         break
        # print("Epoch %s: takes %s seconds" % (epoch, (time.time() - epoch_start_time)))
        # evaluate_model(model,
        #                classidier,
        #                scheduler,
        #                trainloader,
        #                validation_loader,
        #                device,
        #                true_train,
        #                predict_train,
        #                running_loss,
        #                epoch,
        #                num_epochs,
        #                epoch_start_time,
        #                "Validation evaluation...")
        # running_loss = 0
        # model.train()
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
              optimizer_lr=0.00001,
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
    'num_epohs_top': 1,
    'num_epohs_total': 5,
    'optimizer_lr': 0.00001,
    'cuda_device': 0,
}

path_to_train_data = os.path.join(os.getcwd(), 'Cars')
processor = DataSetPreprocessor()
features = processor.get_features(path_to_train_data)
features = features.iloc[0:100]
# ['car_brand_cat', 'car_model_cat', 'body_cat', 'color_cat', 'engine_type_cat', 'transmission_cat',
#                                  'rudder_cat', 'price', 'year', 'engine_volume', 'engine_power']
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

print('Num of car_brand_cat: {}'.format(car_brand_cat))
print('Num of car_model_cat: {}'.format(car_model_cat))
print('Num of body_cat: {}'.format(body_cat))
print('Num of color_cat: {}'.format(color_cat))
print('Num of engine_type_cat: {}'.format(engine_type_cat))
print('Num of transmission_cat: {}'.format(transmission_cat))
print('Num of rudder_cat: {}'.format(rudder_cat))


train_features, validation_features = model_selection.train_test_split(features, random_state=0)
model_dir = os.path.join(os.getcwd(), 'models')
path_to_pretrained_resnet = os.path.join(model_dir, 'AvitoAdGenerator_roc_auc_0.921.pth')
for step in range(1, 5):
    train_features, test_features = model_selection.train_test_split(train_features, test_size=0.1)
    train_data = CarDataSet(train_features, path_to_train_data)
    test_data = CarDataSet(test_features, path_to_train_data)
    validation_data = CarDataSet(validation_features, path_to_train_data)
    train_loader = data.DataLoader(train_data, shuffle=True, batch_size=4)
    test_loader = data.DataLoader(test_data, shuffle=False, batch_size=4)
    validation_loader = data.DataLoader(validation_data, shuffle=False, batch_size=4)
    restore_file = run_train(train_loader,
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


