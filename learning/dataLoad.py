#Dataset processing

import torch
import cv2
import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class LPCVDataset(Dataset):
    def __init__(self, df, transforms):
        super(LPCVDataset, self).__init__()
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        img_path, mask_path = self.df.loc[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #cv2.IMREAD_GRAYSCALE
        #cv2.IMREAD_UNCHANGED
        transformed = self.transforms(image=img, mask=mask)
        
        img = transformed['image']
        mask = transformed['mask']
        img = img/255
        img = img.astype('float32')

        img = np.transpose(img, (2,0,1))

        mask_stacked = np.array([mask==0])

        #number of label class is 14
        for i in range(1, 14):
            mask_stacked = np.concatenate([mask_stacked, np.array([mask==i])])
        mask = mask_stacked.astype(int)
        mask = mask.astype('int64')
        
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        
        return img, mask
    

def make_data_df(img_path,gt_path, type="train"):
    #set train data

    
    if type=='train':
        df = pd.DataFrame(columns=['folder_path','image_name', 'extension'])

        for img in os.listdir(img_path):
            info = {
                'folder_path': [img_path],
                #train name example is train_0000 (10 word)
                'image_name': [img[0:10]], 
                'extension': ['png']}
            info_ = pd.DataFrame(data=info)
            df = pd.concat((df, info_))

        df = df.drop_duplicates(subset=['image_name'])

        imgs_train = df.iloc[:,0] + df.iloc[:,1] + '.png'
        imgs_train = imgs_train.reset_index(drop=True)

        labels_train = gt_path + df.iloc[:,1] + '.png'
        labels_train = labels_train.reset_index(drop=True)


        df = pd.concat((imgs_train,labels_train),axis=1)
    
    # val OR test
    else:
        #set val data
        df = pd.DataFrame(columns=['folder_path','image_name'])

        for img in os.listdir(img_path):
            info = {
                'folder_path': [img_path], 
                'image_name': [img[0:8]]
            }
            info_ = pd.DataFrame(data=info)
            df = pd.concat((df, info_))

        df = df.drop_duplicates(subset=['image_name'])

        imgs_val = df.iloc[:,0] + df.iloc[:,1] + '.png'
        imgs_val = imgs_val.reset_index(drop=True)
        labels_val = gt_path + df.iloc[:,1] + '.png'
        labels_val = labels_val.reset_index(drop=True)

        df = pd.concat((imgs_val,labels_val),axis=1)

    return df


def make_data_loder(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last ):

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return data_loader
