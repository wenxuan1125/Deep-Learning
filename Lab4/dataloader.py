import pandas as pd
import torch
from torch.utils import data
import torchvision.transforms as transforms     # data augmentation
import numpy as np
import os
import PIL
from PIL import ImageFile
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('./train_img.csv')
        label = pd.read_csv('./train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('./test_img.csv')
        label = pd.read_csv('./test_label.csv')

        # img.values.shape = label.values.shape = (7025, 1) = (# of testing data, 1)
        # np.squeeze(img.values) =  np.squeeze(label.values) = (7025, ) = (# of testing data, )
        # np.squeeze(a, axis=None): remove axes of length one from a.
        return np.squeeze(img.values), np.squeeze(label.values)

def imageTransformation(mode):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if mode == 'train':
        # torchvision.transforms.Compose(transforms): composes several transforms together
        # torchvision.transforms.ToTensor: converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
        #                                   to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else: 
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])


    return transform

def cropImage(img):
    non_zeros = img.nonzero()   # find indeices of non zero elements, non_zeros -> Tuple with three arrays as the element
                                # non_zeros[0] = array for the row index of non zero elements
                                # non_zeros[1] = array for the column index of non zero elements
                                # non_zeros[1] = array for the channel index of non zero elements
    
    non_zero_rows = [min(np.unique(non_zeros[0])), max(np.unique(non_zeros[0]))]      # the first and the last row with non zero elements
    non_zero_cols = [min(np.unique(non_zeros[1])), max(np.unique(non_zeros[1]))]      # the first and the last column with non zero elements
    

    dim = max(non_zero_rows[1] - non_zero_rows[0] + 1, non_zero_cols[1] - non_zero_cols[0] + 1)
    crop_img = np.zeros((dim, dim, 3), dtype='uint8')

    if non_zero_rows[1] - non_zero_rows[0] + 1 > non_zero_cols[1] - non_zero_cols[0] + 1:
        # rows more than columns
        diff = non_zero_rows[1] - non_zero_rows[0] - non_zero_cols[1] + non_zero_cols[0]
        col_start = int(diff/2)
        cols_num = non_zero_cols[1] - non_zero_cols[0] + 1
        

        crop_img[:, col_start:(col_start+cols_num), :] = img[non_zero_rows[0]:(non_zero_rows[1]+1), non_zero_cols[0]:(non_zero_cols[1]+1), :]

    else:

        # columns more than rows
        diff = non_zero_cols[1] - non_zero_cols[0] - non_zero_rows[1] + non_zero_rows[0]
        row_start = int(diff/2)
        rows_num = non_zero_rows[1] - non_zero_rows[0] + 1
        

        crop_img[row_start:(row_start+rows_num), :, :] = img[non_zero_rows[0]:(non_zero_rows[1]+1), non_zero_cols[0]:(non_zero_cols[1]+1), :]


    return crop_img


class RetinopathyLoader(data.Dataset):
    # custermize dataset
    # need to implement __init__(), __len__() and __getitem__()
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode

        self.transform = imageTransformation(mode)
        # self.transform = transforms.Compose([transforms.ToTensor()])
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = os.path.join(self.root, self.img_name[index] + '.jpg')
        label = self.label[index]
        img = PIL.Image.open(path)      # PIL image
        img = self.transform(img)       # tensor, shape = (C, H, W)
        
        return img, label

    
