
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms


class ViewDataset(Dataset):
    def __init__(self, root='original/GBA_panoramic/', city='', district='', sub='', length=None, set_size=0):
        """
        Args:
            root+city+district+sub (string): Directory with all the images.
            length (int, optional): Optional num. of images.
            set_size (optional): Set size if need to resize images.
        """

        super(ViewDataset, self).__init__()

        self.path = root+city
        if district:
            self.path += '/'+district
        if sub:
            self.path += '/'+sub
        self.path += '/images'

        self.size = set_size
        try:
            self.file_list = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        except:
            self.file_list = []
        
        if length:
            if length < len(self.file_list):
                self.file_list = self.file_list[:length]


    def __len__(self):
        return self.file_list.__len__()
    
    def __getitem__(self, index):
        self.image = Image.open(self.file_list[index])
        if self.size:
            self.image = self.image.resize((self.size, self.size), Image.LANCZOS)
        self.image = transforms.ToTensor()(self.image) # unit8 -> float, [0,255] -> [0,1]
        self.image = torch.FloatTensor(self.image)
        return self.image
