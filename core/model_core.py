# -*- coding: utf-8 -*-
from PIL import Image
from .ctpn.text_detect import text_detect
from .ctpn.load_tf_model import load_tf_model
from .crnn.recognize import load_crnn_model, recognize_char
from . import image_tools


class ModelCore(object):

    def __init__(self):
        self.sess, self.net = load_tf_model('core/model_data/checkpoints')
        self.crnn_model, self.char_set = load_crnn_model('core/model_data/pytorch-crnn.pth', 'core/model_data/char_std_5990.txt')

    # 模型接口函数
    def ocr(self, img_base64, adjust=False):
        img = image_tools.read_cv2image_from_base64(img_base64)

        text_recs, img_framed, img = text_detect(img, self.sess, self.net)
        results = recognize_char(img, text_recs, self.crnn_model, self.char_set, adjust)

        results = [result[1] for result in results]
        img_url = image_tools.image_to_url(Image.fromarray(img_framed))

        return results, img_url

