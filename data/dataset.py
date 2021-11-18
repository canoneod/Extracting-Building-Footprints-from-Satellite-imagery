from glob import glob
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import os

class Spacenet(torch.utils.data.Dataset):
  def __init__(self, root_dir='/content/gdrive/MyDrive/SpaceNet/processed', train = True, transform=None, target_transform=None):
    self.root_dir = root_dir # processed, train folder in root directory
    self.train = train # whether train or test
  
    if train: # train dir
      data_dir = 'train'
      target_dir = 'train_label_mask'
    else: # test dataset 
      data_dir = 'test'
      target_dir = 'test_label'

    self.transform = transform # only apply transform to labels
    self.target_transform = target_transform

    self.dataList = glob(os.path.join(self.root_dir ,data_dir, '*.npy'))  
    self.targetList = glob(os.path.join(self.root_dir ,target_dir, '*.npy'))
    self.dataList.sort()
    self.targetList.sort()
  
  def RandomCrop(self, image, mask):
  
    i,j,h,w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
    image = TF.crop(image, i,j,h,w)
    mask = TF.crop(mask, i,j,h,w)

    return image, mask

  def __len__(self): # num of images
    return len(self.dataList)

  def __getitem__(self,idx):  # get img target
    # tensor does not support np.uint16. load as np.int64 -> type conversion issue in model(tensor not supported)
    img, target = np.load(self.dataList[idx]).astype(np.float32), np.load(self.targetList[idx]).astype(np.int64)

    
    img, target = TF.to_tensor(img), TF.to_tensor(target)
    img, target = self.RandomCrop(img, target)
    
    # random crop 
    return img.float(), target