import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor, device
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from dataloader import read_bci_data
import argparse
import time
import copy

class EEGNet(nn.Module):
    # custermize our neural network block -> inherit class "nn.Module" and implement method "forward"
    # backward will be automatically implemented when "forward" is implemented
    # layers with weights to be learned are put in constructor "__init__()" 
    # layers without weights to be learned can be put in constructor "__init__()" or "forward"
    def __init__(self, activation: nn.modules.activation):  # parameter "activation" is expected to be type "nn.modules.activation"
        super(EEGNet, self).__init__()  # call constructor of nn.Module

        # torch.nn.Conv2d(in_channels of input image, out_channels produced by the convolution, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        #       groups = number of blocked connections from input channels to output channels
        #       input = (N,C,H,W) or (C,H,W) = (# of data, # of channels of input image, height of image, width of image)
        #       output = (N,C,H,W) or (C,H,W) = (# of data, # of channels of output image, height of image, width of image)
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        #       num_features = C from an expected input of size (N,C,H,W)
        #       input = (N,C,H,W) or (C,H,W)
        #       output = (N,C,H,W) or (C,H,W)
        # torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
        #       kernel_size = the size of the pooling window
        #       input = (N,C,H,W) or (C,H,W)
        #       output = (N,C,H,W) or (C,H,W)
        # torch.nn.Dropout(p=0.5, inplace=False)
        #       p = probability of an element to be zeroed
        # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        #       in_features = size of each input sample
        #       out_features = size of each output sample
        # torch.nn.Flatten(start_dim=1, end_dim=- 1)
        #       Flattens a contiguous range of dims into a tensor. For use with Sequential
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),  
            nn.BatchNorm2d(16)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),  
            nn.BatchNorm2d(32),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),  
            nn.BatchNorm2d(32),
            activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(736, 2)
        )

    def forward(self, x) -> Tensor:
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        output = self.classify(x)

        return output

class DeepConvNet(nn.Module):
    # custermize our neural network block -> inherit class "nn.Module" and implement method "forward"
    # backward will be automatically implemented when "forward" is implemented
    # layers with weights to be learned are put in constructor "__init__()" 
    # layers without weights to be learned can be put in constructor "__init__()" or "forward"
    def __init__(self, activation: nn.modules.activation):  # parameter "activation" is expected to be type "nn.modules.activation"
        super(DeepConvNet, self).__init__()  # call constructor of nn.Module

        # torch.nn.Conv2d(in_channels of input image, out_channels produced by the convolution, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        #       groups = number of blocked connections from input channels to output channels
        #       input = (N,C,H,W) or (C,H,W) = (# of data, # of channels of input image, height of image, width of image)
        #       output = (N,C,H,W) or (C,H,W) = (# of data, # of channels of output image, height of image, width of image)
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        #       num_features = C from an expected input of size (N,C,H,W)
        #       input = (N,C,H,W) or (C,H,W)
        #       output = (N,C,H,W) or (C,H,W)
        # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        #       kernel_size = the size of the pooling window
        #       input = (N,C,H,W) or (C,H,W)
        #       output = (N,C,H,W) or (C,H,W)
        # torch.nn.Dropout(p=0.5, inplace=False)
        #       p = probability of an element to be zeroed
        # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        #       in_features = size of each input sample
        #       out_features = size of each output sample
        # torch.nn.Flatten(start_dim=1, end_dim=- 1)
        #       Flattens a contiguous range of dims into a tensor. For use with Sequential
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), padding='valid', bias=False),  
            nn.Conv2d(25, 25, kernel_size=(2, 1), padding='valid', bias=False),
            nn.BatchNorm2d(25),
            activation,
            nn.MaxPool2d(1, 2),
            nn.Dropout(p=0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), padding='valid', bias=False),
            nn.BatchNorm2d(50),
            activation,
            nn.MaxPool2d(1, 2),
            nn.Dropout(p=0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), padding='valid', bias=False),
            nn.BatchNorm2d(100),
            activation,
            nn.MaxPool2d(1, 2),
            nn.Dropout(p=0.5)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), padding='valid', bias=False),
            nn.BatchNorm2d(200),
            activation,
            nn.MaxPool2d(1, 2),
            nn.Dropout(p=0.5)
        )
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8800, 2)
        )

    def forward(self, x) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        output = self.classify(x)

        return output






def train(model, device, train_loader, test_loader, epochs, lr, print_interval, arguments, now_time):

    if model == 'EEGNet':
        models = { 'relu': EEGNet(nn.ReLU()).to(device), 'leaky_relu': EEGNet(nn.LeakyReLU()).to(device), 'elu': EEGNet(nn.ELU()).to(device)}
        # print(models[1])
    elif model == 'DeepConvNet':
        models = { 'relu': DeepConvNet(nn.ReLU()).to(device), 'leaky_relu': DeepConvNet(nn.LeakyReLU()).to(device), 'elu': DeepConvNet(nn.ELU()).to(device)}

    train_accuracy ={'relu': [], 'leaky_relu': [], 'elu': []}
    train_loss ={'relu': [], 'leaky_relu': [], 'elu': []}
    test_accuracy ={'relu': [], 'leaky_relu': [], 'elu': []}
    # test_loss ={'relu': [], 'leaky_relu': [], 'elu': []}


    
    for key, model in models.items():

        # torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
        #       params =  iterable of parameters to optimize or dicts defining parameter groups
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_test_accruracy = 0

        for current_epoch in range(epochs):
            
            # train model

            # set the model in training mode 
            # inform layers such as Dropout and BatchNorm, which are designed to behave differently during training and testing
            model.train()
            total_train_correct = 0
            total_test_correct = 0
            total_train_loss = 0

            for datas, labels in train_loader:

                datas = datas.to(device)
                labels = labels.to(device).long()  # type of labels has to be 'long'
                predicts = model(datas)
                # print(type(predicts))
                loss = torch.nn.CrossEntropyLoss()(predicts, labels)

                optimizer.zero_grad()       # clear previous gradient
                loss.backward()             # compute gradient
                optimizer.step()            # update parameters of model with gradient

                total_train_loss+=loss.item()                                                 # Tensor.item() -> tensor to scalar
                total_train_correct += (predicts.max(dim=1)[1] == labels).sum().item()        # predicts.max(dim=1)[1] = id of maximum value, predicts.max(dim=1)[0] = maximum value
                                                                                        
            total_train_loss /= len(train_loader.dataset)
            total_train_correct = 100.* total_train_correct / len(train_loader.dataset)

            train_accuracy[key].append(total_train_correct)
            train_loss[key].append(total_train_loss)

            # test model
            # set the model in testing mode 
            # inform layers such as Dropout and BatchNorm, which are designed to behave differently during training and testing
            model.eval()
            with torch.no_grad():
                for datas, labels in test_loader:
                    datas = datas.to(device)
                    labels = labels.to(device).long()  # type of labels has to be 'long'
                    predicts = model(datas)
                                                                    # Tensor.item() -> tensor to scalar
                    total_test_correct += (predicts.max(dim=1)[1] == labels).sum().item()

                total_test_correct = 100.* total_test_correct / len(test_loader.dataset)    

            test_accuracy[key].append(total_test_correct)

            if(arguments.save_model):
                if total_test_correct > best_test_accruracy:
                    # save the model
                    
                    best_test_accruracy = total_test_correct
                    best_model_weights = copy.deepcopy(model.state_dict())
                    path = './checkpoints/' + str(best_test_accruracy) + '_' + arguments.model + '_' + key + '_' + str(lr) + '_' + str(arguments.batch_size) + '_' + str(current_epoch) + '_' + now_time + '.pth'
                    torch.save(best_model_weights, path)

            if current_epoch % print_interval == print_interval - 1:
                print(f'Activation: {key}, Epoch: {current_epoch}, Train Loss: {total_train_loss}, Train Accuracy: {total_train_correct}, Test Accuracy: {total_test_correct}')
                
    return train_accuracy, train_loss, test_accuracy

def demo(device, test_loader):
    model = EEGNet(nn.ReLU()).to(device)
    path = './demo_model.pth'
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():
        total_test_correct = 0
        for datas, labels in test_loader:
            datas = datas.to(device)
            labels = labels.to(device).long()  # type of labels has to be 'long'
            predicts = model(datas)
                                                            # Tensor.item() -> tensor to scalar
            total_test_correct += (predicts.max(dim=1)[1] == labels).sum().item()

        total_test_correct = 100.* total_test_correct / len(test_loader.dataset)    
    print(f'Test Accuracy: {total_test_correct}')


def show_result(epoch,train_accuracy_dic, test_accuracy_dic, train_loss_dic, model):

    # plt.subplot(1, 2, 1)
    
    plt.title(model)
    for activation_function, accuracy in train_accuracy_dic.items():
        plt.plot(range(epoch), accuracy, label=activation_function+'_train')
    for activation_function, accuracy in test_accuracy_dic.items():
        print(f'{activation_function}: {max(accuracy)}')
        plt.plot(range(epoch), accuracy, label=activation_function+'_test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.title('Loss')
    # for activation_function, loss in train_loss_dic.items():
    #     plt.plot(range(epoch), loss)
    plt.show()

def training_arguments():
    # EEGNet -> lr 0.001
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='EEGNet', type=str,help='EEGNet or DeepConvNet')
    parser.add_argument("--activation", default='relu', type=str, help='relu, leaky_relu or elu')
    parser.add_argument("--batch_size", default=64, type=int, help='batch size')
    parser.add_argument("--lr", default=0.001, type=float, help='learning rate')
    parser.add_argument("--loss_function", default='cross_entropy', type=str, help='cross_entropy or ...')
    parser.add_argument("--optimizer", default='adam', type=str, help='adam or ...')
    parser.add_argument("--epochs", default=300, type=int, help='training epochs')
    parser.add_argument("--print_interval", default=10, type=int,help='print interval')
    parser.add_argument("--load_model", dest='load_model', action='store_true')
    parser.add_argument('--no_load', dest="load_model", action='store_false')
    parser.add_argument("--save_model", dest='save_model', action='store_true')
    parser.add_argument('--no_save', dest="save_model", action='store_false')
    parser.set_defaults(load_model=False)
    parser.set_defaults(save_model=False)
    args = parser.parse_args()
    return args

def main():
    # arguments used in training
    now_time = str(int(time.time()))
    arguments = training_arguments()
    model = arguments.model
    epochs = arguments.epochs
    lr = arguments.lr
    batch_size = arguments.batch_size
    # optimizer = arguments.optimizer
    loss_function = arguments.loss_function
    print_interval = arguments.print_interval

    # read data 
    train_data, train_label, test_data, test_label = read_bci_data()    # np.ndarray  
                                                                        # train_data.shape = (1080, 1, 2, 750) = (batch, 1, channel of brainwave, time), train_label.shape = (1080,)
                                                                        # test_data.shape = (1080, 1, 2, 750), test_label.shape = (1080,)
    
    # torch.utils.data.TensorDataset(*tensors): warpping tensors as a dataset
    # Each sample will be retrieved by indexing tensors along the first dimension -> works like "zip" in python
    # dataset[i] is a tuple, dataset[i][0] = the ith data in Tensor, dataset[i][1] = the ith label in Tensor
    train_dataset = TensorDataset(Tensor(train_data), Tensor(train_label))
    test_dataset = TensorDataset(Tensor(test_data), Tensor(test_label))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if(arguments.load_model):
        # only test, for demo
        demo(device, test_loader)

        
    else: 
        train_accuracy, train_loss, test_accuracy = train(model, device, train_loader, test_loader, epochs, lr, print_interval, arguments, now_time)
        show_result(epochs,train_accuracy, test_accuracy, train_loss, model)
        


if __name__ == '__main__':
    main()