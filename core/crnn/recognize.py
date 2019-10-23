# -*- coding: utf-8 -*-
import cv2
from math import *
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from . import crnn


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


def load_crnn_model(cpt_path, char_set_path):
    char_set_lines = open(char_set_path, 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set_lines[1:]] + ['卍'])

    n_class = len(char_set)
    crnn_model = crnn.CRNN(32, 1, n_class, 256)
    crnn_model.load_state_dict(torch.load(cpt_path, map_location=torch.device('cpu')))

    return crnn_model, char_set


def sort_box(box):
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])): min(ydim - 1, int(pt3[1])),
             max(1, int(pt1[0])): min(xdim - 1, int(pt3[0]))]

    return imgOut


def decode(preds, char_set):
    pred_text = ''
    for i in range(len(preds)):
        if preds[i] != 0 and ((i == 0) or (i != 0 and preds[i] != preds[i - 1])):
            pred_text += char_set[int(preds[i]) - 1]

    return pred_text


def predict(img, model, char_set):
    (w, h) = img.size
    size_h = 32
    ratio = size_h / float(h)
    size_w = int(w * ratio)

    transform = resizeNormalize((size_w, size_h))
    image = transform(img)
    image = image.unsqueeze(0)

    model.eval()
    preds = model(image)
    preds = preds.max(2)[1]
    preds = preds.squeeze()
    pred_text = decode(preds, char_set)

    return pred_text


def recognize_char(img, text_recs, model, char_set, adjust=False):
    text_recs = sort_box(text_recs)

    results = []
    xDim, yDim = img.shape[1], img.shape[0]

    for rec in text_recs:
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])

        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

        partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)

        if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
            continue

        image = Image.fromarray(partImg).convert('L')
        text = predict(image, model, char_set)

        if len(text) > 0:
            results.append(((rec[0], rec[1], rec[6], rec[7]), text))

    return results
