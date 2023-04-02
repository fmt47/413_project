import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class COVIDDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.data = []
        self.transform = transform
        self.map = {'negative': 0, 'positive': 1}
        
        with open(data_file, 'r') as f:
            for line in f:
                patient_id, filename, label, data_source = line.strip().split(' ')
                image_path = os.path.join(os.path.splitext(data_file)[0], filename)
                self.data.append((image_path, self.map[label]))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
class GANDataset(Dataset):
    """Two folders: corona and normal, each consists of 500 images"""
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform
        self.map = {'corona': 1, 'normal': 0}
        
        # iterate through corona and normal folders
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                self.data.append((image_path, self.map[folder]))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label