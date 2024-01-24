import os
import random
import argparse
import torch
from tqdm import tqdm
from dataloader import load_train_data, load_test_data
from model import DDPM, ContextUnet
from matplotlib.animation import FuncAnimation, PillowWriter
from evaluator import evaluation_model
from torchvision.utils import make_grid, save_image
from statistics import mean 


def parse_args():
    parser = argparse.ArgumentParser()

	## What to do
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test" , action="store_true")
    parser.add_argument('--load', action='store_true')

    parser.add_argument("--seed", default=1, type=int, help="manual seed")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--optimizer", default="adam", help="optimizer to train with")
    parser.add_argument("--epochs", type=int, default=600, help="number of epochs to train for")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--beta1", default=1e-4, type=float, help="beta1 for adam optimizer")
    parser.add_argument("--beta2", default=0.02, type=float, help="beta2 for adam optimizer")
    # 
    parser.add_argument("--input_dim", default=64, type=int, help="dimension of input image")
    parser.add_argument("--n_feature", default=128, type=int, help="number of features")
    parser.add_argument("--n_class", default=24, type=int, help="number of classes")
    parser.add_argument("--n_T", default=400, type=int, help="number of T")

    ## Paths 
    parser.add_argument("--model_dir", default="./checkpoint", help="base directory to save your model checkpoints") 
    parser.add_argument("--model_name", default="epoch590.pth", help="base directory to save your model checkpoints") 
    parser.add_argument("--data_root", default="./dataset", help="root directory for data") 
    parser.add_argument("--result_dir", default="./result", help="base directory to save predicted images") 
    parser.add_argument("--log_dir", default="./log", help="base directory to save training logs") 
    parser.add_argument("--exp_name", default="DDPM") 
    parser.add_argument("--test_file", default="test.json")

    args = parser.parse_args()
    return args

def train(args, train_loader, test_loader, new_test_loader, model, optimizer, evaluator, device):

    if(args.load):
        model.load_state_dict(torch.load("{}/{}".format(args.model_dir, args.model_name)))
    best_eval_acc = 0
    best_gen_img = None

    for epoch in range(args.epochs):
        model.train()
        loss_ema = None
        loss_list = []
        for train_imgs, train_labels in tqdm(train_loader):

            # imgs.shape = (128, 3, 64, 64) = (B, C, W, H)
            # labels.shape = (128, 24) = (B, #types)
            # print(imgs.shape)
            # print(labels.shape)

            # linear lrate decay
            optimizer.param_groups[0]['lr'] = args.lr*(1-epoch/args.epochs)

           
            optimizer.zero_grad()
            train_imgs = train_imgs.to(device)
            train_labels = train_labels.to(device)
            # labels = labels.to(torch.int64)
            # print(labels)
            # print(labels)
            loss = model(train_imgs, train_labels)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            
            loss_list.append(loss_ema)

            optimizer.step()

        

        if epoch % 5 == 0 or epoch == int(args.epochs - 1):

            model.eval()
            for test_labels in test_loader:
                with torch.no_grad():
                    
                    
                    test_labels = test_labels.to(device)
                    imgs_gen, imgs_gen_store = model.sample(test_labels.shape[0], test_labels, (3, 64, 64), device)
                    eval_acc = evaluator.eval(imgs_gen, test_labels)

                    # if eval_acc > best_eval_acc:
                    #     best_eval_acc = eval_acc
                    #     best_gen_img = imgs_gen

                        #torch.save(model.state_dict(), "{}/epoch{}_{}.pth".format(args.model_dir, epoch, best_eval_acc))
                    save_image(imgs_gen, "{}/train/test_epoch{}_{}.png".format(args.result_dir, epoch, eval_acc), normalize=True)
                        # print('saved model at ' + save_dir + f"model_{epoch}.pth")

            for new_test_labels in new_test_loader:
                with torch.no_grad():
                    
                    
                    new_test_labels = new_test_labels.to(device)
                    new_imgs_gen, new_imgs_gen_store = model.sample(new_test_labels.shape[0], new_test_labels, (3, 64, 64), device)
                    new_eval_acc = evaluator.eval(new_imgs_gen, new_test_labels)

                    # if eval_acc > best_eval_acc:
                    #     best_eval_acc = eval_acc
                    #     best_gen_img = imgs_gen

                        
                    save_image(new_imgs_gen, "{}/train/new_test_epoch{}_{}.png".format(args.result_dir, epoch, new_eval_acc), normalize=True)
                        # print('saved model at ' + save_dir + f"model_{epoch}.pth")

            torch.save(model.state_dict(), "{}/epoch{}_{}_{}.pth".format(args.model_dir, epoch, eval_acc, new_eval_acc))
            print(f'Epoch: {epoch}, Loss: {mean(loss_list)}, Test Accuracy: {eval_acc}, New Test Accuracy: {new_eval_acc}')
        else: 
            print(f'Epoch: {epoch}, Loss: {mean(loss_list)}')

        # ## Save generated images
        # save_image(x_gen, "{}/test/gen.png".format(args.result_dir), normalize=True)

        # # optionally save model
        # if save_model and epoch == int(args.epochs - 1):
            

            


def test(args, test_loader, model, evaluator, device):
    model.load_state_dict(torch.load("{}/{}".format(args.model_dir, args.model_name)))
    
    with torch.no_grad(): 
        for test_labels in tqdm(test_loader):   
            # labels.shape = (32, 24) = (#test data, #types)
            # print(labels.shape)

            # fixed_noise = torch.randn(labels.shape[0], self.args.z_dim, 1, 1, device=self.device)
            # imgs_gen, x_gen_store = model.sample(labels.shape[0], (3, 64, 64), device, guide_w=w)
            test_labels = test_labels.to(device)
            imgs_gen, imgs_gen_store = model.sample(test_labels.shape[0], test_labels, (3, 64, 64), device)

        # print(imgs_gen.shape)
        # print(imgs_gen_store.shape)
            eval_acc = evaluator.eval(imgs_gen, test_labels)
        ## Save generated images
        # save_image(imgs_gen, "{}/{}/pred_{}-{}.png".format(self.args.result_dir, self.args.exp_name, epoch, step), normalize=True)
        save_image(imgs_gen, "{}/test/gen.png".format(args.result_dir), normalize=True)


        print(f'Accuracy: {eval_acc}')
            

def main():
    args = parse_args() 
    ## Set random seed for reproducibility
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    ## Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##################
	## Load dataset ##
	##################
    if args.train:
        train_loader, test_loader, new_test_loader = load_train_data(args, device)
    elif args.test:
        test_loader = load_test_data(args, device)  
    else:
        raise ValueError("Mode [train/test] not determined!")

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=args.n_feature, n_classes=args.n_class), betas=(args.beta1, args.beta2), n_T=args.n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=args.lr)

    evaluator = evaluation_model()
    

    if args.train:
        train(args, train_loader, test_loader, new_test_loader, ddpm, optimizer, evaluator, device)
    elif args.test: 
        test(args, test_loader, ddpm, evaluator, device)

if __name__ == '__main__':
    main()