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

def imageTransformation():
    
    transform = transforms.Compose([
        transforms.Resize(512)
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

def main():
    img_name, label = getData('train')
    root ='./train_image/'
    save_root = './preprocess/train/'
    transform = imageTransformation()
    length = len(label)
    for i in range(15662,length):
        print(f'{i}/{length}')
        path = os.path.join(root, img_name[i] + '.jpeg')
        img = PIL.Image.open(path)      # PIL image
        img = np.array(img)
        
        img = cropImage(img)
        img = PIL.Image.fromarray(img)
        img = transform(img)       
        img.save(save_root + img_name[i] + '.jpg')

if __name__ == '__main__':
    main()