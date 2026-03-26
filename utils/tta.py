import torch
import torchvision.transforms as T

tta_transforms = [
    T.Compose([T.Resize((224,224)), T.ToTensor()]),
    T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(p=1), T.ToTensor()]),
    T.Compose([T.Resize((224,224)), T.RandomRotation(15), T.ToTensor()])
]

def tta_predict(model, image, device):
    preds = []

    for tfm in tta_transforms:
        img = tfm(image).unsqueeze(0).to(device)
        pred = torch.sigmoid(model(img))
        preds.append(pred)

    return torch.stack(preds).mean(0)