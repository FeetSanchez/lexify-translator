from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import torch

app = FastAPI(title="Lexify Translator")

# Modelo público e gratuito
model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

class TextInput(BaseModel):
    text: str

@app.post("/translate")
def translate_text(item: TextInput):
    inputs = tokenizer(item.text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    output = tokenizer.decode(translated[0], skip_special_tokens=True)
    return {"translated_text": output}
