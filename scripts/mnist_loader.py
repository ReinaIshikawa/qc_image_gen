# ============================
# loader for mnist dataset
# https://qiita.com/taiga10969/items/24d85860ffcd724de9eb
# ============================


import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CustomMNISTDataset(Dataset):
    """
    a custom dataset loader for MNIST dataset
    """
    def __init__(self, root='./data', train=True, transform=None, download=True, size=16):
        """
        Args:
            root: the directory to save the dataset
            train: True for training data, False for test data
            transform: the transformation to apply to the image
            download: whether to download the dataset
            size: the target size for resizing the image (default: 16)
        """
        self.size = size
        self.dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            transform=transform if transform is not None else transforms.ToTensor(),
            download=download
        )
    
    def __len__(self):
        return len(self.dataset)

    def change_image_to_halftone(self, image, threshold=0.4):
        image = image > threshold
        return image.float()
    
    def resize_image(self, image, size):
        """
        Resize image to the specified size
        Args:
            image: tensor of shape (1, H, W)
            size: target size (int or tuple)
        Returns:
            resized image tensor of shape (1, size, size)
        """
        if isinstance(size, int):
            size = (size, size)
        # F.interpolate expects (batch, channels, height, width)
        image = image.unsqueeze(0)  # (1, 1, H, W)
        image = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
        image = image.squeeze(0)  # (1, size, size)
        return image
    
    def __getitem__(self, idx):
        # image is a tensor of shape (1, 28, 28)
        image, label = self.dataset[idx]
        # resize image to specified size
        if self.size != 28:  # Only resize if size is different from original
            image = self.resize_image(image, self.size)
        # halftone transformation -> binary image (float of 0 or 1)
        image = self.change_image_to_halftone(image)
        return image, label
