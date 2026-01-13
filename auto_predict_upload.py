
import requests

LS_URL = "http://localhost:8080"
LS_API_TOKEN = "55a38a7ca2c097f613b0314c0269e002d7f72724"
PROJECT_ID = 1
MODEL_URL = "http://localhost:8000/predict"

headers = {
    "Authorization": f"Token {LS_API_TOKEN}"
}

try:
    tasks_resp = requests.get(f"{LS_URL}/api/projects/{PROJECT_ID}/tasks", headers=headers)
    tasks_resp.raise_for_status()
    tasks = tasks_resp.json()
except Exception as e:
    print("获取任务失败:", e)
    exit(1)

for task in tasks:
    task_id = task["id"]
    print(f"Processing task {task_id}")

    payload = {"tasks": [task]}
    try:
        model_resp = requests.post(MODEL_URL, json=payload)
        model_resp.raise_for_status()
        prediction = model_resp.json()
    except Exception as e:
        print(f"调用模型接口失败: {e}")
        continue

    if "result" not in prediction:
        print(f"  ⚠️ No result returned for task {task_id}")
        continue

    prediction_payload = {
        "model_version": "v1",
        "result": prediction["result"]
    }

    try:
        pred_resp = requests.post(
            f"{LS_URL}/api/projects/{PROJECT_ID}/tasks/{task_id}/predictions",
            headers=headers,
            json=prediction_payload
        )
        pred_resp.raise_for_status()
        print(f"  ✅ Prediction saved for task {task_id}")
    except Exception as e:
        print(f"  ❌ Failed to save prediction for task {task_id}: {pred_resp.status_code} {pred_resp.text}")
