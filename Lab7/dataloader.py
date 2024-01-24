import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class iclevrDataset(Dataset):
    def __init__(self, args, device, mode="train"):
        self.args = args
        self.mode = mode
        self.device = device
        self.data_root = args.data_root

        self.object_idx = json.load(open("{}/objects.json".format(self.data_root)))

        if self.mode == "train":
            data_dict = json.load(open("{}/train.json".format(self.data_root)))
            self.data_list = list(data_dict.items())
        elif self.mode == "test":
            self.data_list = json.load(open("{}/test.json".format(self.data_root)))
        elif self.mode == "new_test":
            self.data_list = json.load(open("{}/new_test.json".format(self.data_root)))

        self.transforms = transforms.Compose([
            transforms.Resize(self.args.input_dim), 
            transforms.CenterCrop(self.args.input_dim),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.mode == "train":
            ## Read image
            img_arr = Image.open("{}/iclevr/{}".format(self.data_root, self.data_list[index][0])).convert("RGB")
            img_tensor = self.transforms(img_arr)
            img_tensor = img_tensor

            ## Condition: object type
            cond_onehot = torch.zeros(len(list(self.object_idx.keys())))
            for object_type in self.data_list[index][1]:
                cond_idx = self.object_idx[object_type]
                cond_onehot[cond_idx] = 1

            return img_tensor, cond_onehot
        elif self.mode == "test":
			## Condition
            cond_onehot = torch.zeros(len(list(self.object_idx.keys())))
            for object_type in self.data_list[index]:
                cond_idx = self.object_idx[object_type]
                cond_onehot[cond_idx] = 1 

            return cond_onehot

        elif self.mode == "new_test":
			## Condition
            cond_onehot = torch.zeros(len(list(self.object_idx.keys())))
            for object_type in self.data_list[index]:
                cond_idx = self.object_idx[object_type]
                cond_onehot[cond_idx] = 1 

            return cond_onehot

def load_train_data(args, device):
    """Load i-CLEVR Dataset"""
    print("\nBuilding training & testing dataset...")
    
    train_dataset = iclevrDataset(args, device, "train")
    test_dataset  = iclevrDataset(args, device, "test")
    new_test_dataset  = iclevrDataset(args, device, "new_test")

    print("# training samples: {}".format(len(train_dataset)))
    print("# testing  samples: {}".format(len(test_dataset)))
    print("# new testing  samples: {}".format(len(new_test_dataset)))


    train_loader = DataLoader(
        train_dataset, 
        num_workers=args.num_workers, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        num_workers=args.num_workers, 
        # batch_size=args.batch_size, 
        batch_size = len(test_dataset), 
        shuffle=False
    )

    new_test_loader = DataLoader(
        new_test_dataset, 
        num_workers=args.num_workers, 
        # batch_size=args.batch_size, 
        batch_size = len(new_test_dataset), 
        shuffle=False
    )

    return train_loader, test_loader, new_test_loader

def load_test_data(args, device):
    print("\nBuilding testing dataset...")

    test_dataset = iclevrDataset(args, device, "test")

    print("# testing  samples: {}".format(len(test_dataset)))

    test_loader = DataLoader(
        test_dataset, 
        num_workers=args.num_workers, 
        batch_size=len(test_dataset), 
        shuffle=False
    )
    
    return test_loader

def load_new_test_data(args, device):
    print("\nBuilding new testing dataset...")

    test_dataset = iclevrDataset(args, device, "new_test")

    print("# new testing  samples: {}".format(len(test_dataset)))

    new_test_loader = DataLoader(
        test_dataset, 
        num_workers=args.num_workers, 
        batch_size=len(test_dataset), 
        shuffle=False
    )
    
    return new_test_loader