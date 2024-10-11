import torch
import pandas as pd
from torch.utils.data import Dataset
import glob
import os
from PIL import Image

class ImageSequenceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        if type(csv_file) == list:
            data = pd.DataFrame()
            for file in csv_file:
                df = pd.read_csv(file)
                data = pd.concat([data, df])
            self.dataframe_data = data.reset_index(drop=True)
        elif type(csv_file) == str:
            self.dataframe_data = pd.read_csv(csv_file)
            
        self.transform = transform

    def __len__(self):
        return len(self.dataframe_data)

    def __getitem__(self, idx):
        img_folder_path = self.dataframe_data['image_path'].iloc[idx]
        img_files = sorted(glob.glob(os.path.join(img_folder_path, '*.jpg')))
        
        images = []
        for i in range(0, len(img_files), 2):
            image = Image.open(img_files[i])
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        images = torch.stack(images, dim=0) 
        label = self.dataframe_data['label'].iloc[idx]
        
        additional_data = self.dataframe_data.iloc[idx].drop(['image_path', 'label']).to_dict()
        
        return images, label, additional_data 
