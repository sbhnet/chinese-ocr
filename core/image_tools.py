# -*- coding: utf-8 -*-
import os
import re
import io
import base64
import numpy as np
import cv2
from PIL import Image, ExifTags, ImageDraw, ImageFont


def url_to_img(img_url):
    img_base64 = re.sub('^data:image/.+;base64,', '', img_url)
    return Image.open(io.BytesIO(base64.b64decode(img_base64.encode())))


def get_base64_from_file(path):
    with open(path, 'rb') as file:
        return str(base64.b64encode(file.read()), encoding='utf-8')


def read_image_from_base64(img_base64):
    image = Image.open(io.BytesIO(base64.b64decode(img_base64.encode())))

    # 自动旋转图像
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except:
        pass

    if not image.mode == 'RGB':
        image = image.convert('RGB')
    return image


def read_cv2image_from_base64(img_base64):
    return np.array(read_image_from_base64(img_base64))


def image_to_url(img):
    output_buffer = io.BytesIO()
    img.save(output_buffer, format='PNG')
    return 'data:image/png;base64,' + str(base64.b64encode(output_buffer.getvalue()), encoding='utf-8')


def cv2image_to_url(img):
    image = Image.fromarray(np.uint8(img))
    return image_to_url(image)


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords1 (xyxy) from img1_shape to img0_shape
    gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
    coords[:, [0, 2]] -= (img1_shape[1] - img0_shape[1] * gain) / 2  # x padding
    coords[:, [1, 3]] -= (img1_shape[0] - img0_shape[0] * gain) / 2  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords


def img_resize(image, new_size=1024):
    img_w, img_h = image.size
    if img_w > new_size:
        ratio = new_size / img_w
        image = image.resize((int(ratio * img_w), int(ratio * img_h)), Image.ANTIALIAS)
    elif img_h > new_size:
        ratio = new_size / img_h
        image = image.resize((int(ratio * img_w), int(ratio * img_h)), Image.ANTIALIAS)
    return image


def cv2img_resize(image, new_size=1024):
    img_h, img_w, _ = image.shape
    if img_w > new_size:
        ratio = new_size / img_w
        image = cv2.resize(image, (int(ratio * img_w), int(ratio * img_h)))
    elif img_h > new_size:
        ratio = new_size / img_h
        image = cv2.resize(image, (int(ratio * img_w), int(ratio * img_h)))
    return image


def letterbox(img, new_shape=416, color=(127.5, 127.5, 127.5), mode='auto'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


def draw_box_and_label(image, box, label):
    cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)

    # cv2直接打印中文为乱码，需要转换成PIL格式再绘制
    pilimg = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    text_size = 24
    try:
        font = ImageFont.truetype('wqy-zenhei.ttc', text_size)
    except:
        os.system('apt-get install -y ttf-wqy-zenhei')  # 需要安装中文字体
        font = ImageFont.truetype('wqy-zenhei.ttc', text_size)

    x1 = box[0]
    y1 = box[1] - text_size - 3
    if y1 < 0:
        y1 = 0
    point = (x1, y1)
    draw.rectangle((x1, y1, x1 + text_size * len(label), y1 + text_size), fill=(255, 0, 0))
    draw.text(point, label, (255, 255, 255), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

    # PIL图片转cv2 图片
    return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)


def draw_gps_info(image, gps_info):
    if gps_info is not None:
        # cv2直接打印中文为乱码，需要转换成PIL格式再绘制
        pilimg = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_w, img_h = pilimg.size
        draw = ImageDraw.Draw(pilimg)  # 图片上打印
        text_size = 24
        try:
            font = ImageFont.truetype('wqy-zenhei.ttc', text_size)
        except:
            os.system('apt-get install -y ttf-wqy-zenhei')  # 需要安装中文字体
            font = ImageFont.truetype('wqy-zenhei.ttc', text_size)

        label = '%s  %s' % (gps_info[4], gps_info[3])
        draw.text((0, img_h - text_size -3), label, (255, 0, 0), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

        # PIL图片转cv2 图片
        return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    else:
        return image
