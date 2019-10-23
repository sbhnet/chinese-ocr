# -*- coding:utf-8 -*-
from ucumos.wrapped import load_packaged_model
from core import image_tools


model = load_packaged_model('./out/中文OCR')  # 从模型文件夹加载模型对象
model.init_model_data.inner('')  # 初始化模型数据

img_base64 = image_tools.get_base64_from_file('test_data/name_card.jpg')
response = model.ocr.inner(img_base64)  # 调用模型接口内部方法
print(response)

import json
response = json.loads(response)
image_tools.url_to_img(response['img_url']).save('test_result.png')
print(response['results'])
