import os
import torch 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class c_dataset(Dataset):
    
    def __init__(self, path, transform=None):
         
        self.path = path
        self.transform = transform
        self.img_list = os.path.join(self.path, 'images')
        self.labels_list = os.path.join(self.path, 'labels')
           
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.path, self.img_list[idx])
        img = Image.open(img_path)
        label_f = open(os.path.join(self.path, self.labels_list[idx]), 'r')
        label_s = label_f.readline()[0]
        
        if label_s[0] == '0':
            label = [0, None]
        else:
            label = [1, [label_s[1:5]], [label_s[5:]]]
        
        if self.transform is None:
            img = transforms.ToTensor(img)
            
        img = self.transform(img)
        label = transforms.ToTensor(img)
        
        return img, label
    
    