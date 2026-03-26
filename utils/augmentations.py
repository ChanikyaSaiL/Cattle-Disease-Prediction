import torchvision.transforms as T
from configs.config import IMAGE_SIZE

def get_train_transforms():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        T.ToTensor(),
    ])

def get_val_transforms():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])