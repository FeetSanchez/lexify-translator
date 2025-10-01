from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()
# redeploy trigger

# Token será configurado nas variáveis do Railway
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "Helsinki-NLP/opus-mt-en-pt"  # modelo de tradução en->pt
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
def translate(request: TranslationRequest):
    payload = {"inputs": request.text}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    
    if response.status_code != 200:
        return {"error": response.text}
    
    result = response.json()
    # Estrutura esperada: [{'translation_text': '...'}]
    if isinstance(result, list) and "translation_text" in result[0]:
        return {"translation": result[0]["translation_text"]}
    return {"error": "Unexpected response format", "details": result}

@app.get("/")
def home():
    return {"status": "online", "message": "Lexify Translator API running via Hugging Face Inference"}
