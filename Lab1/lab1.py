import numpy as np
import matplotlib.pyplot as plt
import argparse


class GenerateData:
    @staticmethod                               # represent that the following function can be called without construct an instance of this class
    def _generate_linear(n=100):                # _FunctionName: '_' -> private function
        pts = np.random.uniform(0, 1, (n, 2))   # (low, high, size)
        inputs = []
        labels = []

        for pt in pts:
            inputs.append([pt[0], pt[1]])
            # distance = (pt[0], pt[1])/1.414
            if(pt[0] > pt[1]):
                labels.append(0)
            else:
                labels.append(1)
        
        return np.array(inputs), np.array(labels).reshape(n, 1)

    @staticmethod  
    def _generate_XOR_easy():
        inputs = []
        labels = []

        for i in range(11):
            inputs.append([0.1*i, 0.1*i])
            labels.append(0)

            if 0.1*i == 0.5:
                continue

            inputs.append([0.1*i, 1-0.1*i])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape(21, 1)

    @staticmethod 
    def generate_data(mode, n=100):             # FunctionName: no '_' -> public function
        if mode == 'linear':
            return GenerateData._generate_linear(n)
        elif mode == 'xor':
            return GenerateData._generate_XOR_easy()

class NeuralNet:
    def __init__(self, hidden_width=(5,5), lr=0.01, max_train_epochs=5000, print_interval=100, activation=True):
        self.hidden_width = hidden_width
        self.lr = lr
        self.max_train_epochs = max_train_epochs
        self.print_interval = print_interval
        self.activation = activation
        # print(self.hidden_width, self.lr, self.max_train_epochs, self.print_interval)
        # print(self.activation)
        '''
        x->(w1)->a1->(sig1)->z1->(w2)->a2->(sig2)->z2->(w3)->a3->(sig3)->out==y 
        '''
        np.random.seed(3)
        self.x = np.zeros((1, 2))
        self.w1 = np.random.rand(2, hidden_width[0])
        self.w2 = np.random.rand(hidden_width[0], hidden_width[1])
        self.w3 = np.random.rand(hidden_width[1], 1)

        self.a1 = np.zeros((1, hidden_width[0]))
        self.a2 = np.zeros((1, hidden_width[1]))
        self.a3 = np.zeros((1, 1))

        self.z1 = np.zeros((1, hidden_width[0]))
        self.z2 = np.zeros((1, hidden_width[1]))
        self.z3 = np.zeros((1, 1))
        self.loss = np.zeros((1, 1))


    def forward(self, x):
        # x.shape = (1, 2)
        self.x[0, 0] = x[0]
        self.x[0, 1] = x[1]
        self.a1 = self.x@self.w1
        if(self.activation):
            self.z1 = sigmoid(self.a1)
        else:
            self.z1 = self.a1

        self.a2 = self.z1@self.w2

        if(self.activation):
            self.z2 = sigmoid(self.a2)
        else:
            self.z2 = self.a2
        self.a3 = self.z2@self.w3
        if(self.activation):
            self.z3 = sigmoid(self.a3)
        else: 
            self.z3 = self.a3
        out = self.z3

        
        return out
    
    def backward(self, pred_y, y):
        self.loss[0] = (pred_y - y) * (pred_y - y)                  # min square error
        # pred_y == self.z3
        
        grad_loss_z3 =  -2 * (y - self.z3)                          # shape = (1, 1)
        
        if(self.activation):
            grad_z3_a3 = derivative_sigmoid(self.z3)                # shape = (1, 1)
        else:
            grad_z3_a3 = np.ones(self.z3.shape)
        grad_loss_a3 = grad_loss_z3 * grad_z3_a3                    # shape = (1, 1)

        grad_a3_w3 = self.z2                                        # shape = (1, h1)
        grad_loss_w3 = grad_a3_w3.T @ grad_loss_a3                  # shape = (h1, 1)
        
        grad_a3_z2 = self.w3                                        # shape = (h1, 1)
        grad_loss_z2 = grad_loss_a3 @ grad_a3_z2.T                  # shape = (1, h1)


        if(self.activation):
            grad_z2_a2 = derivative_sigmoid(self.z2)                # shape = (1, h1)
        else:
            grad_z2_a2 = np.ones(self.z2.shape)
        grad_loss_a2 = grad_loss_z2 * grad_z2_a2                    # shape = (1, h1)

        grad_a2_w2 = self.z1                                        # shape = (1, h0)
        grad_loss_w2 = grad_a2_w2.T @ grad_loss_a2                  # shape = (h0, h1)

        grad_a2_z1 = self.w2                                        # shape = (h0, h1)
        grad_loss_z1 = grad_loss_a2 @ grad_a2_z1.T                  # shape = (1, h0)

        if(self.activation):
            grad_z1_a1 = derivative_sigmoid(self.z1)                # shape = (1, h0)
        else:
            grad_z1_a1 = np.ones(self.z1.shape)
        grad_loss_a1 = grad_loss_z1 * grad_z1_a1                    # shape = (1, h0)

        grad_a1_w1 = self.x                                         # shape = (1, 2)
        grad_loss_w1 = grad_a1_w1.T @ grad_loss_a1                  # shape = (2, h0)

        # update weights
        self.w1 -= self.lr * grad_loss_w1
        self.w2 -= self.lr * grad_loss_w2
        self.w3 -= self.lr * grad_loss_w3


    def train(self, datas, labels):

        data_num = datas.shape[0]
        loss = []
        epochs = []
        
        
        for epoch in range(self.max_train_epochs):
            epoch_loss = 0

            for i in range(data_num):
                data = datas[i]
                label = labels[i]

                out = self.forward(data)
                self.backward(out, label)

                epoch_loss += self.loss

            loss.append((epoch_loss/data_num)[0,0])
            epochs.append(epoch)
 
            if epoch % self.print_interval == (self.print_interval - 1):
                print(f'Epoch: {epoch:6d}, loss: {(epoch_loss/data_num)[0,0]}') 

            if epoch_loss == 0:
                break

        print('Training finished')

        return loss, epochs

    def test(self, datas, labels):
        data_num = datas.shape[0]
        right = 0
        loss = 0
        predict_labels = np.zeros((data_num, 1))
        for i in range(data_num):
            data = datas[i]
            label = labels[i,0]

            out = self.forward(data)[0,0]

            if out > 0.5:
                predict_labels[i] = 1
            else:
                predict_labels[i] = 0
            
            right += (label == predict_labels[i])[0]
            loss += (label-out)*(label-out)

            print(f'Data{i}:    Ground truth: {label}, Prediction: {out:.5f}')

        print(f'Accuracy: {right/data_num} ({right}/{data_num}), Loss: {loss}')
        

        return predict_labels
def show_learning_curve(epoch,loss):
    plt.title('Learning Curve')
    plt.plot(epoch, loss)
    plt.show()

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if(y[i] == 0):
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict Result', fontsize=18)
    for i in range(x.shape[0]):
        if(pred_y[i] == 0):
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    # the input of this function should be the output of the sigmoid function
    return np.multiply(x, 1 - x)    # multiply arguments element-wise -> not matrix multiplication
def train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='linear', type=str,help='linear or xor')
    parser.add_argument("--hidden_width", default=[10, 10], type=int, nargs='+', help='hidden width')
    parser.add_argument("--lr", default=0.01, type=float, help='learning rate')
    parser.add_argument("--max_train_epochs", default=6000, type=int, help='maximum train epochs',)
    parser.add_argument("--print_interval", default=100, type=int,help='print interval')
    parser.add_argument("--activation", dest='activation', action='store_true')
    parser.add_argument('--no_activation', dest="activation", action='store_false')
    parser.set_defaults(activation=True)
    args = parser.parse_args()
    return args
def main():
    args = train_options()
    x, y = GenerateData.generate_data(args.dataset, n=100)
    net = NeuralNet(hidden_width=(args.hidden_width[0],args.hidden_width[1]), lr=args.lr, 
        max_train_epochs=args.max_train_epochs, print_interval=args.print_interval, activation = args.activation)
    loss, epochs = net.train(x, y)
    pred_y = net.test(x, y)

    show_learning_curve(epochs, loss)
    show_result(x, y, pred_y)




if __name__ == "__main__":
    main()