import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_checkpoint = "Rogendo/en-sw"
print("Loading model and tokenizer (this takes a few seconds)...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
print("Ready! Type English text and press Enter. Type 'quit' to exit.\n")

while True:
    english_text = input("English: ").strip()
    if english_text.lower() in ("quit", "exit", "q"):
        print("Kwaheri! 👋")
        break
    if not english_text:
        continue

    inputs = tokenizer(english_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        translated_ids = model.generate(**inputs, max_length=512, num_beams=4)

    swahili_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    print(f"Swahili: {swahili_text}\n")