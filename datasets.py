import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import os 
from PIL import Image

# Required constants.
_dataset_dir = '../../Shared/Pin-Han/chest_xray/'

# Training transforms
def get_train_transform(image_size):
    train_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size=(image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.RandomRotation(degrees=(-20,+20)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    ])
    return train_transform

# Validation transforms
def get_valid_transform(image_size):

    valid_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size=(image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) 

    ])
    return valid_transform


class CXR_Dataset(Dataset):

    def __init__(self,path, tfm, files = None):
        super(CXR_Dataset).__init__()
        self.path = path
        self.normal = [os.path.join(path+'/NORMAL',x) for x in os.listdir(path+'/NORMAL') if x.endswith(".jpeg")]
        self.pneumo = [os.path.join(path+'/PNEUMONIA',x) for x in os.listdir(path+'/PNEUMONIA') if x.endswith(".jpeg")]
        self.files = self.normal+self.pneumo
        if files != None:
            self.files = files
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        #im = (np.array(im)/255).astype('uint8')
        im = self.transform(im)
        #split = fname.split(os.sep)
        split = fname.split('/')[-2]
        if split == 'PNEUMONIA':
            label = torch.Tensor([1])
        else:
            label = torch.Tensor([0])
        return im, label
    
def get_datasets(image_size):
    train_set = CXR_Dataset(os.path.join(_dataset_dir,'train'), tfm=get_train_transform(image_size))
    val_set = CXR_Dataset(os.path.join(_dataset_dir,'val'), tfm=get_valid_transform(image_size))
    return train_set, val_set   
        

def get_data_loaders(dataset_train, dataset_valid, batch_size, num_workers):
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader