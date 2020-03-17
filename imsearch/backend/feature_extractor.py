from PIL import Image
import base64
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms


def get_extractor(arch='resnet50'):
    model_ft = models.__dict__[arch](pretrained=True)
    extractor = nn.Sequential(*list(model_ft.children())[:-1])
    return extractor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = get_extractor('resnet50').to(device)
extractor.eval()
image_transforms = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(244),
    transforms.ToTensor(),
])


def extract_features(img):
    img = image_transforms(Image.fromarray(img))
    img = img.unsqueeze_(0).to(device)

    with torch.no_grad():
        out = extractor(img).squeeze().detach().cpu()
    return base64.b64encode(out.numpy()).decode("utf-8")
