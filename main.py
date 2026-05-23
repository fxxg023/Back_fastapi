from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import uuid
import shutil
from pathlib import Path
import requests
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import io
import json
import cv2
import base64
from peft import PeftModel

# 1. 创建应用
app = FastAPI(title="图像识别", version="1.0")

origins = [
    "http://localhost:6006",
    "https://u862097-afef-4e584c3c.bjb1.seetacloud.com:8443",
]

# 2. 跨域配置（允许你的 Vue 前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 生产可改为你的前端地址 http://localhost:5173
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. 临时目录保存上传图片
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 4. 请求体结构（URL识别）
class RecognizeURLRequest(BaseModel):
    url: str
    highQualityMode: bool = True

print("正在加载模型...")
# 模型配置
MODEL_NAME = "../Qwen2-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和处理器
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model,"../lora")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
print("模型加载完成！")

def get_bbox(result_str):
    try:
        # 1. 解析 JSON 字符串
        data = json.loads(result_str)
        
        # 2. 取出 bbox
        bbox = data.get("bbox", [])
        
        # 3. 确保返回的是列表
        return bbox if isinstance(bbox, list) else []
    
    except json.JSONDecodeError:
        # JSON 格式错误（比如你刚才的 unterminated string）
        return []
    except Exception:
        # 其他意外错误
        return []

def draw_bbox(target,ipath):
    img = cv2.imread(ipath)
    #print(target)
    if(len(target)!=4):
        print("bbox 格式错误")
        return

    x1,y1,x2,y2=target
    h,w=img.shape[:2]
    if(x1<=1):
        x1=round(x1*w)
        y1=round(y1*h)
        x2=round(x2*w)
        y2=round(y2*h)
    else:
        x1 = round(x1 * w / 1000)
        y1 = round(y1 * h / 1000)
        x2 = round(x2 * w / 1000)
        y2 = round(y2 * h / 1000)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    #filename = f"{uuid.uuid4()}.jpg"
    #save_path = f"static/{filename}"
    #cv2.imwrite(save_path, img)
    print(f"画框完成！")
    imb,img_encoded = cv2.imencode('.jpg', img)
    base64_str = base64.b64encode(img_encoded).decode('utf-8')

    return base64_str
    #image_url = f"http://127.0.0.1:8000/static/{filename}"
    #return image_url

# -------------------
# 【核心】本地大模型推理函数
# -------------------
def local_model_recognize(image_path: str = None, image_url: str = None, question: str = ""):
    try:
        # 1. 加载图片
        if image_path:
            image = Image.open(image_path).convert("RGB")
        elif image_url:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(requests.utils.get_fileobj(response)).convert("RGB")
        else:
            raise ValueError("必须提供图片路径或URL")

        #with open(image_path,"rb") as f:
        #    image_data = f.read()

        # 2. 构建提示词
        messages = [
            {
                "role": "system",
                "content":[
                    {"type": "text",
                     "text":'你是一个结构化输出助手。每次回答必须严格输出固定格式，禁止输出 markdown、解释、多余文字。格式为：{"answer": "你的回答", "bbox": [x1, y1, x2, y2]}'
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        # 3. 预处理
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        # 4. 生成结果
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        #print(output_text)
        target_bbox=get_bbox(output_text)
        out_image=draw_bbox(target_bbox,image_path)
        
        # 5. 解析结果（简单解析，你可以根据需要优化）
        return {
            "code": 0,
            "data": {
                "image":out_image,
                "similarity": 0.90,
                "answer": output_text
            }
        }

    except Exception as e:
        print(f"模型推理错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型推理失败：{str(e)}")

# -------------------
# 接口1：上传图片识别
# -------------------
@app.post("/api/recognize/upload")
async def recognize_upload(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        # 校验文件类型
        # allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
        # if file.content_type not in allowed_types:
        #    raise HTTPException(status_code=400, detail="不支持的图片格式")

        # 保存文件
        filename = f"{uuid.uuid4()}_{file.filename}"
        save_path = UPLOAD_DIR / filename
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print("开始推理")
        # 调用本地大模型
        result = local_model_recognize(
            image_path=str(save_path),
            question=question.strip()
        )
        print("推理完成")
        os.remove(save_path)
        return result
        #with open(f"{file.filename}","wb") as buffer:
        #    shutil.copyfileobj(file.file,buffer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败：{str(e)}")

# -------------------
# 接口2：URL图片识别
# -------------------
@app.post("/api/recognize/url")
async def recognize_url(data: RecognizeURLRequest):
    try:
        # 调用本地大模型
        result = local_model_recognize(
            image_url=data.url,
            high_quality=data.highQualityMode
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL识别失败：{str(e)}")

# 根路径
@app.get("/")
async def root():
    return {"message": "图像识别后端运行成功"}

# 启动时自动清理临时文件
@app.on_event("shutdown")
def cleanup():
    for file in UPLOAD_DIR.glob("*"):
        try:
            os.remove(file)
        except:
            pass