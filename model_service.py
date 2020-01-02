# -*- coding: utf-8 -*-
import os
from core.model_core import ModelCore


class ModelService(object):

    # 声明model_core对象并赋空置。必须存在，不要修改或删除！！！
    def __init__(self):
        self.model_core = None

    # 从文件中加载模型数据，例如预训练好的模型参数等。必须存在，不要修改或删除！！！
    def init_model_data(self):
        if self.model_core is None:
            try:
                self.model_core = ModelCore()
            except:
                self.model_core = None

    # 模型接口函数
    def ocr(self, img_base64):
        return self.model_core.ocr(img_base64)


if __name__ == "__main__":

    model_service = ModelService()
    model_service.init_model_data()

    from core import image_tools

    img_base64 = image_tools.get_base64_from_file('test_data/name_card.jpg')
    print(img_base64)

    results, img_url = model_service.ocr(img_base64)
    image_tools.url_to_img(img_url).save('test_result.png')
    print(results)
