# -*- coding:utf-8 -*-
import json
from ucumos.modeling import Model
from ucumos.session import AcumosSession
from ucumos.metadata import Requirements
from model_service import ModelService


# 创建模型服务对象
model_service = ModelService()


# 封装模型API接口函数： 初始化模型数据。该接口函数仅供系统调用，必须存在，不要修改或删除！！！
def init_model_data(text: str) -> str:
    model_service.init_model_data()
    return 'model_data_loaded'


# 封装模型API接口函数
def ocr(img_base64: str) -> str:
    results, img_url = model_service.ocr(img_base64)
    return json.dumps({"results": results, "img_url": img_url}, ensure_ascii=False)


# 打包模型文件
model = Model(init_model_data=init_model_data,
              ocr=ocr)
session = AcumosSession()
reqs = Requirements(reqs=['PIL', 'cv2', 'tensorflow==1.3.0', 'torch', 'torchvision', 'easydict'],
                    req_map={'PIL': 'pillow', 'cv2': 'opencv-python'},
                    scripts=['./model_service.py'],
                    packages=['./core'])
session.dump(model, '中文OCR', './out/', reqs)
