import os
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root1, root2, transform=None):
        self.root1 = root1
        self.root2 = root2
        self.transform = transform

        self.files1 = sorted(os.listdir(root1))
        self.files2 = sorted(os.listdir(root2))

    def __len__(self):
        return len(self.files1)

    def __getitem__(self, idx):

        img_file1 = self.files1[idx]
        img_file2 = self.files2[idx]

        img_path_touch = os.path.join(self.root1, self.files1[idx])
        image_touch = Image.open(img_path_touch)
        img_path_vision = os.path.join(self.root2, self.files2[idx])
        image_vision = Image.open(img_path_vision)
        if self.transform:
            image_touch = self.transform(image_touch)
            image_vision = self.transform(image_vision)

        return image_touch, image_vision, img_file1, img_file2