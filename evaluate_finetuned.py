import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Load the FINE-TUNED model
model_path = "./finetuned-en-sw-final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Same exact evaluation slice as your baseline
dataset = load_dataset("michsethowusu/english-swahili_sentence-pairs", split="train[:200]")
metric = evaluate.load("sacrebleu")

predictions = []
references = []

print("Running fine-tuned evaluation...")
for example in tqdm(dataset):
    inputs = tokenizer(example["English"], return_tensors="pt",
                       truncation=True, max_length=512).to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_length=512, num_beams=4)
    predictions.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    references.append([example["Swahili"]])

results = metric.compute(predictions=predictions, references=references)
print(f"\n✅ Fine-Tuned BLEU Score: {results['score']:.2f}")
print(f"   Baseline BLEU Score:    25.44")
print(f"   Change:                {results['score'] - 25.44:+.2f}")