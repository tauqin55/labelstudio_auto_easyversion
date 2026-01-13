from fastapi import FastAPI, Request
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
import os

app = FastAPI()

# 直接加载官方预训练模型（不微调）
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name)
model.eval()
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Label Studio 配置的标签必须大小写完全匹配
id2label = {0: "Normal", 1: "Benign", 2: "Malignant"}

@app.get("/predict/health")
def health_check():
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/setup")
async def setup(request: Request):
    return {"status": "ready"}

# 路径映射：容器 -> 本地
CONTAINER_ROOT = "/data/upload"
LOCAL_ROOT = "D:/labelstudio_data/media/upload"

def container_to_local_path(container_path):

    # 处理 /data/upload 路径（对应本地 D:/labelstudio_datasets）
    if container_path.startswith("/data/upload"):
        relative_path = container_path[len("/data/upload"):].lstrip("/").replace("/", "\\")
        return os.path.join("D:\\labelstudio_data\\media\\upload", relative_path)
    else:
        return container_path

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    print("Received request data:", data)

    try:
        task = data["tasks"][0]
        image_path_in_container = task["data"]["image"]
        print(f"Processing image path: {image_path_in_container}")
        image_path_local = container_to_local_path(image_path_in_container)

        print(f"Image path resolved to: {image_path_local}")
        if not os.path.exists(image_path_local):
            print("Image not found!")
            return {"result": []}

        image = Image.open(image_path_local).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()

        label = id2label.get(pred, "Unknown")
        print(f"Predicted label: {label}")

        return {
            "result": [
                {
                    "from_name": "classification",
                    "to_name": "image",
                    "type": "choices",
                    "value": {
                        "choices": [label]
                    }
                }
            ]
        }

    except Exception as e:
        print("Prediction error:", str(e))
        return {"result": []}
