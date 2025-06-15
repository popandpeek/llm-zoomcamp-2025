import torch
import time
import psutil
from transformers import AutoModel, AutoTokenizer

MODELS = ['intfloat/e5-large-v2', 'intfloat/e5-base-v2']

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on device: {device}")

# Test embedding inference
sample_text = "Led development of a scalable data pipeline using Airflow and Kafka."

for MODEL_NAME in MODELS:
    ram_before = psutil.virtual_memory().used / (1024 ** 3)
    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    end_load = time.time()

    ram_after = psutil.virtual_memory().used / (1024 ** 3)
    print(f"Model load time: {end_load - start_load:.2f} sec")
    print(f"RAM used: {ram_after - ram_before:.2f} GB")

    inputs = tokenizer(sample_text, return_tensors="pt").to(device)

    start_infer = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end_infer = time.time()

    print(f"Inference time: {end_infer - start_infer:.4f} sec")
    print(f"Output shape: {outputs.last_hidden_state.shape}")
