
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class ViewDataset(Dataset):
    def __init__(self, root='original_datas/shenzhen_panoramic/', length=None, start_idx=0,set_size=0):
        super(ViewDataset, self).__init__()

        view_map = pd.read_csv(root+'shenzhen.csv',header=None)
        view_map.columns = ['file_name','name','Lon','Lat'] #指定表头
        view_map['file_name'] = view_map['file_name'].apply(lambda x: x + ".jpg") #为文件名添加后缀

        self.root = root
        self.size = set_size
        if length:
            self.file_list = view_map['file_name'][start_idx:start_idx+length] #只截取需要的部分
            self.file_list = self.file_list.reset_index(drop=True)
        else:    
            self.file_list = view_map['file_name'] #全部读入

    def __len__(self):
        return self.file_list.__len__()
    
    def __getitem__(self, index):
        self.file_list
        self.image = Image.open(self.root+self.file_list[index])
        if self.size:
            self.image = self.image.resize((self.size, self.size), Image.LANCZOS) #缩放
        self.image = transforms.ToTensor()(self.image) # unit8 -> float, [0,255] -> [0,1]
        self.image = torch.FloatTensor(self.image)
        return self.image
