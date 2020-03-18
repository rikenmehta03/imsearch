from __future__ import division

import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image

from .models import Darknet
from .utils import load_classes, pad_to_square, resize, non_max_suppression, rescale_boxes, download_files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.path.join(os.environ.get('HOME'), '.imsearch', 'yolo')


class Detector(object):
    model = None

    def __init__(self):
        download_files()

        self.img_size = 416
        self.config_path = os.path.join(DATA_PATH, 'yolov3.cfg')
        self.weights_path = os.path.join(DATA_PATH, 'yolov3.weights')
        self.class_path = os.path.join(DATA_PATH, 'coco.names')
        self.conf_thres = 0.8
        self.nms_thres = 0.4
        if Detector.model is None:
            model = Darknet(self.config_path,
                            img_size=self.img_size).to(device)
            model.load_darknet_weights(self.weights_path)
            model.eval()
            Detector.model = model
        self.classes = load_classes(self.class_path)

    def _load_image(self, img):
        actual_size = img.shape
        img = self._process_image(img)
        img = img.unsqueeze_(0)
        return img, actual_size
    
    def _process_image(self, img):
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        img, _ = pad_to_square(img, 0)
        img = resize(img, self.img_size)
        return img

    def predict(self, img):
        input_img, actual_shape = self._load_image(img)
        input_img = input_img.to(device)

        with torch.no_grad():
            detections = Detector.model(input_img)
            detections = non_max_suppression(
                detections, self.conf_thres, self.nms_thres)[0]

        response = []
        if detections is not None:
            detections = rescale_boxes(
                detections, self.img_size, actual_shape[:2])
            for x1, y1, x2, y2, _, cls_conf, cls_pred in detections:
                x1 = max(min(int(x1.item()), actual_shape[1]-1), 0)
                x2 = max(min(int(x2.item()), actual_shape[1]-1), 0)
                y1 = max(min(int(y1.item()), actual_shape[0]-1), 0)
                y2 = max(min(int(y2.item()), actual_shape[0]-1), 0)
                response.append({
                    'box': [x1, y1, x2, y2],
                    'score': cls_conf.item(),
                    'label': int(cls_pred),
                    'name': self.classes[int(cls_pred)]
                })
        return response
