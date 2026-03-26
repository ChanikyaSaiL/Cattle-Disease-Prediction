import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class CattleDataset(Dataset):
    def __init__(self, base_dir, csv_path, transform=None):
        self.base_dir = base_dir
        self.transform = transform

        self.df = pd.read_csv(csv_path)

        self.image_names = self.df.iloc[:, 0].values
        self.labels = self.df.iloc[:, 1:].values.astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        # IMPORTANT: path includes train/valid/test prefix
        img_path = os.path.join(self.base_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label