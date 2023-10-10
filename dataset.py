import np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


# Augmentation
def rotate(img, msk, degrees=(-15,15), p=0.5):
    if torch.rand(1) < p:
        degree = np.random.uniform(*degrees)
        img = img.rotate(degree, Image.NEAREST)
        msk = msk.rotate(degree, Image.NEAREST)
    return img, msk

def horizontal_flip(img, msk, p=0.5):
    if torch.rand(1) < p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
    return img, msk

def vertical_flip(img, msk, p=0.5):
    if torch.rand(1) < p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            msk = msk.transpose(Image.FLIP_TOP_BOTTOM)
    return img, msk

def augment(img, msk):
    img, msk = horizontal_flip(img, msk)
    img, msk = vertical_flip(img, msk)
    img, msk = rotate(img, msk)
    return img, msk


# Dataset
class myData(Dataset):
    """Data path saved in .npz file
        x_train, x_test: (256, 256, 3), [0, 255]
        y_train, y_test: (256, 256), {0, 255} """
        
    def __init__(self, data_path, type=None, transform=False):
      super().__init__()

      data_np = np.load(data_path)
      self.images = data_np[f"x_{type}"]
      self.masks  = data_np[f"y_{type}"]
      self.transform = transform

    def __getitem__(self, idx):
      img = Image.fromarray(self.images[idx])
      msk = Image.fromarray(self.masks[idx])

      if self.transform:
          img, msk = augment(img, msk)

      img = transforms.ToTensor()(np.array(img))
      msk = np.expand_dims(msk, axis = -1)
      msk = transforms.ToTensor()(np.array(msk))
      
      return img, msk

    def __len__(self):
      return len(self.images)
