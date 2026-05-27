# Back_fastapi

毕设后端
本项目为基于Fastapi框架开发的后端程序，用于完成接收前端字符串与图像、调用模型处理、返回结果等工作

## 快速启动

本项目使用但不强制要求Qwen2-7B-VL-Instruct模型，这里并未提供模型，模型需自行下载，并且可自行更换
下载好模型后，调整代码中的模型地址与模型名称

自行选择使用conda或其它虚拟环境管理工具，本项目采用conda

下载本代码并创建虚拟环境后，在终端安装所需库函数，执行
`pip install -r requirements.txt`

然后通过
```
uvicorn main:app --reload
这里的main是指文件名main.py,将其映射为app
```
启动后端服务器

后续可以通过地址
`http://127.0.0.1:8000/docs`
访问测试网页