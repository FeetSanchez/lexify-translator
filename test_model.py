from transformers import pipeline

translator = pipeline("translation_en_to_pt", model="Helsinki-NLP/opus-mt-en-pt")
print(translator("Hello World!"))