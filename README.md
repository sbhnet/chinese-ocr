# 文本识别

本模型改编自GitHub开源项目：[https://github.com/ooooverflow/chinese-ocr](https://github.com/ooooverflow/chinese-ocr)

原项目是一款基于CTPN（tensorflow）+CRNN（pytorch）+CTC的不定长文本检测和识别。


## API接口

该模型实现了1个API接口，其调用形式和返回值格式分别如下：

- ocr：

    - HTTP方法：POST

    - 模型方法：ocr

        HTTP请求体格式：
        
        {
        
          “img_base64”: <压缩图像的base64编码字符串>
          
        }
        
        HTTP响应体格式：
        
        {
        
          “value": <JSON字符串>
          
        }
        
        其中JSON字符串中JSON对象的格式如下：
        
        {
        
          "results": [<文本1>, <文本2>, ......], 
              
          "img_url": <带文本框定的原始图像的base64编码URL字符串>
        }

## 模型托管和演示

### 模型打包及导入

1. 运行model_pack.py，将在out文件夹下生成一个压缩文件：中文文本识别.zip。

2. 进入CubeAI平台“模型导入”界面（[https://cubeai.dimpt.com/#/ucumos/onboarding](https://cubeai.dimpt.com/#/ucumos/onboarding)），将上述生产的zip文件导入CubeAI平台。

### 模型托管

- https://cubeai.dimpt.com/#/ucumos/solution/e6d7e4afd22c427db664967d3cc45c75/view
    
### 模型能力开放

- https://cubeai.dimpt.com/#/ai-ability/ability/a20d770f667d4ed19fb2a709b6afdc00/view
    
### 模型演示

- https://cubeai.dimpt.com/udemo/#/chinese-ocr
 