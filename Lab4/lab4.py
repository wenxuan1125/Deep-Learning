import torch
from dataloader import RetinopathyLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.nn as nn
from model import ResNet50, ResNet18
import argparse
from tqdm import tqdm
import time
import copy
from sklearn.metrics import confusion_matrix
import seaborn as sn

def train(model_name, device, train_loader, test_loader, epochs, lr, print_interval, now_time, save_model=False, load_model=False):
    
    if load_model:
        if model_name == 'resnet18':
            path = './checkpoints/resnet18.pth'

        elif model_name == 'resnet50':
            path = './checkpoints/resnet50.pth'
            pretrained = torchvision.models.resnet50()
            in_features = pretrained.fc.in_features
            pretrained.fc = nn.Linear(in_features, 5)
            pretrained = pretrained.to(device)
            pretrained.load_state_dict(torch.load(path))

            models = { 'pretrained': pretrained}
            train_accuracy = {'pretrained': []}
            train_loss = {'pretrained': []}
            test_accuracy = {'pretrained': []}

    else:
        if model_name == 'resnet18':
            pretrained = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            in_features = pretrained.fc.in_features
            pretrained.fc = nn.Linear(in_features, 5)
            pretrained = pretrained.to(device)

            not_pretrained = ResNet18().to(device)

            models = { 'pretrained': pretrained, 'not_pretrained': not_pretrained}
            # print(models[1])
        elif model_name == 'resnet50':
            pretrained = torchvision.models.resnet50(weights='IMAGENET1K_V2')
            in_features = pretrained.fc.in_features
            pretrained.fc = nn.Linear(in_features, 5)
            pretrained = pretrained.to(device)
            
            not_pretrained = ResNet50().to(device)
            models = { 'pretrained': pretrained, 'not_pretrained': not_pretrained}

        train_accuracy = {'pretrained': [], 'not_pretrained': []}
        train_loss = {'pretrained': [], 'not_pretrained': []}
        test_accuracy = {'pretrained': [], 'not_pretrained': []}


    for key, model in models.items():

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        loss_function = nn.CrossEntropyLoss()
        best_test_accruracy = 0

        for current_epoch in range(epochs):
            model.train()
            total_train_correct = 0
            total_test_correct = 0
            total_train_loss = 0

            for datas, labels in tqdm(train_loader):
                datas = datas.to(device)
                labels = labels.to(device).long()  # type of labels has to be 'long'
                predicts = model(datas)

                loss = loss_function(predicts, labels)

                optimizer.zero_grad()       # clear previous gradient
                loss.backward()             # compute gradient
                optimizer.step()            # update parameters of model with gradient

                total_train_loss+=loss.item()                                                 # Tensor.item() -> tensor to scalar
                total_train_correct += (predicts.max(dim=1)[1] == labels).sum().item()        # predicts.max(dim=1)[1] = id of maximum value, predicts.max(dim=1)[0] = maximum value

            total_train_loss /= len(train_loader.dataset)
            total_train_correct = 100.* total_train_correct / len(train_loader.dataset)  

            train_accuracy[key].append(total_train_correct)
            train_loss[key].append(total_train_loss)
            # print(datas.shape, labels.shape)


            model.eval()
            with torch.no_grad():
                for datas, labels in tqdm(test_loader):
                    datas = datas.to(device)
                    labels = labels.to(device).long()  # type of labels has to be 'long'
                    predicts = model(datas)
                                                                    # Tensor.item() -> tensor to scalar
                    total_test_correct += (predicts.max(dim=1)[1] == labels).sum().item()

                total_test_correct = 100.* total_test_correct / len(test_loader.dataset)    

            test_accuracy[key].append(total_test_correct)

            if(save_model):
                if total_test_correct > best_test_accruracy:
                    # save the model
                    best_test_accruracy = total_test_correct
                    best_model_weights = copy.deepcopy(model.state_dict())
                    path = './checkpoints/' + str(best_test_accruracy) + '_' + model_name + '_' + key + '_' + str(current_epoch) + '_' + now_time + '.pth'
                    torch.save(best_model_weights, path)
                elif current_epoch == epochs - 1:
                    end_weights = copy.deepcopy(model.state_dict())
                    path = './checkpoints/' + str(total_test_correct) + '_' + model_name + '_' + key + '_' + str(current_epoch) + '_' + now_time + '.pth'
                    torch.save(end_weights, path)



            if current_epoch % print_interval == print_interval - 1:
                print(f'{key} {model_name} Epoch: {current_epoch}, Train Loss: {total_train_loss}, Train Accuracy: {total_train_correct}, Test Accuracy: {total_test_correct}')

    return train_accuracy, train_loss, test_accuracy

def test(model_name, device, test_loader):
    
    if model_name == 'resnet18_pretrain':
        # path = './checkpoints/resnet18.pth'
        # model = ResNet18().to(device)

        path = './checkpoints/resnet18_pretrain.pth'
        model = torchvision.models.resnet18()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 5)
        model = model.to(device)
    elif model_name == 'resnet18':
        path = './checkpoints/resnet18.pth'
        model = ResNet18().to(device)
    else:
        path = './checkpoints/resnet50.pth'
        model = ResNet50().to(device)

        # path = './checkpoints/resnet50_pretrain.pth'
        # model = torchvision.models.resnet50()
        # in_features = model.fc.in_features
        # model.fc = nn.Linear(in_features, 5)
        # model = model.to(device)

    model.load_state_dict(torch.load(path))
    model.eval()
    ground_labels, predict_labels = [], []
    with torch.no_grad():
        total_test_correct = 0
        for datas, labels in tqdm(test_loader):
            datas = datas.to(device)
            labels = labels.to(device).long()  # type of labels has to be 'long'
            predicts = model(datas)
                                                            # Tensor.item() -> tensor to scalar
            total_test_correct += (predicts.max(dim=1)[1] == labels).sum().item()
            ground_labels.extend(labels.detach().cpu().numpy().tolist())
            predict_labels.extend(predicts.max(dim=1)[1].detach().cpu().numpy().tolist())

        total_test_correct = 100.* total_test_correct / len(test_loader.dataset)    
    print(f'Test Accuracy: {total_test_correct}')

    plot_confusion_matrix(ground_labels, predict_labels, title=model_name)
    
def demo(device, test_loader):
    path = './checkpoints/demo.pth'

    model = torchvision.models.resnet50()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 5)
    model = model.to(device)

    model.load_state_dict(torch.load(path))
    model.eval()
    ground_labels, predict_labels = [], []
    with torch.no_grad():
        total_test_correct = 0
        for datas, labels in tqdm(test_loader):
            datas = datas.to(device)
            labels = labels.to(device).long()  # type of labels has to be 'long'
            predicts = model(datas)
                                                            # Tensor.item() -> tensor to scalar
            total_test_correct += (predicts.max(dim=1)[1] == labels).sum().item()
            ground_labels.extend(labels.detach().cpu().numpy().tolist())
            predict_labels.extend(predicts.max(dim=1)[1].detach().cpu().numpy().tolist())

        total_test_correct = 100.* total_test_correct / len(test_loader.dataset)    
    print(f'Test Accuracy: {total_test_correct}')

    plot_confusion_matrix(ground_labels, predict_labels)

def plot_confusion_matrix(y_true, y_pred, classes=[0, 1, 2, 3, 4],
                          title=None,
                          cmap=plt.cm.Blues,
                          filename=None):
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')
    fig, ax = plt.subplots()
    sn.heatmap(cm, annot=True, ax=ax, cmap=cmap, fmt='.2f')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.xaxis.set_ticklabels(classes, rotation=45)
    ax.yaxis.set_ticklabels(classes, rotation=0)
    plt.title(title)
    plt.show()
    # plt.savefig(filename, dpi=300)
def show_result(epoch,train_accuracy_dic, test_accuracy_dic, train_loss_dic, model_name):
    
    plt.title(model_name)
    for pretrained_or_not, accuracy in train_accuracy_dic.items():
        plt.plot(range(epoch), accuracy, label='Train(' + pretrained_or_not+')')
    for pretrained_or_not, accuracy in test_accuracy_dic.items():
        # print(f'{pretrained_or_not}: {max(accuracy)}')
        plt.plot(range(epoch), accuracy, label='Test(' + pretrained_or_not+')')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def training_arguments():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='train', type=str,help='train, test or demo')
    parser.add_argument("--model", default='resnet18', type=str,help='resnet18 or resnet50')
    parser.add_argument("--activation", default='relu', type=str, help='relu, leaky_relu or elu')
    parser.add_argument("--batch_size", default=4, type=int, help='batch size')
    parser.add_argument("--lr", default=0.001, type=float, help='learning rate')
    parser.add_argument("--loss_function", default='cross_entropy', type=str, help='cross_entropy or ...')
    parser.add_argument("--optimizer", default='SGD', type=str, help='adam or ...')
    parser.add_argument("--epochs", default=10, type=int, help='training epochs')
    parser.add_argument("--print_interval", default=1, type=int,help='print interval')
    parser.add_argument("--load_model", dest='load_model', action='store_true')
    parser.add_argument('--no_load', dest="load_model", action='store_false')
    parser.add_argument("--save_model", dest='save_model', action='store_true')
    parser.add_argument('--no_save', dest="save_model", action='store_false')
    parser.set_defaults(load_model=False)
    parser.set_defaults(save_model=False)
    args = parser.parse_args()
    return args

def main():

    arguments = training_arguments()
    now_time = str(int(time.time()))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model_name = arguments.model
    epochs = arguments.epochs
    lr = arguments.lr
    batch_size = arguments.batch_size
    print_interval = arguments.print_interval
    save_model = arguments.save_model
    load_model = arguments.load_model


    train_dataset = RetinopathyLoader('./preprocess/train/', 'train')
    test_dataset = RetinopathyLoader('./preprocess/test/', 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    if(arguments.mode == 'test'):
        # only test
        test(model_name, device, test_loader)
    if(arguments.mode == 'demo'):
        # for demo
        demo(device, test_loader)
    else: 
        train_accuracy, train_loss, test_accuracy = train(model_name, device, train_loader, test_loader, epochs, lr, print_interval, now_time, save_model, load_model)
        show_result(epochs,train_accuracy, test_accuracy, train_loss, model_name)
    
if __name__ == '__main__':
    main()
    # train_dataset = RetinopathyLoader('./train_image/', 'train')
    # test_dataset = RetinopathyLoader('./test_image/', 'test')
    # train_dataset.__getitem__(4)