import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# 1. Load the model and tokenizer
model_checkpoint = "Rogendo/en-sw"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Load a small portion of the evaluation dataset
print("Loading dataset...")
dataset = load_dataset("michsethowusu/english-swahili_sentence-pairs", split="train[:200]")

# 3. Load the SacreBLEU metric
metric = evaluate.load("sacrebleu")

# 4. Generate predictions and collect references
predictions = []
references = []

print("Running baseline evaluation...")
for example in tqdm(dataset):
    english_text = example["English"]
    reference_text = example["Swahili"]

    inputs = tokenizer(english_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_length=512, num_beams=4)

    prediction = tokenizer.decode(ids[0], skip_special_tokens=True)
    predictions.append(prediction)
    references.append([reference_text]) # SacreBLEU expects a list of references

# 5. Compute and print the score
results = metric.compute(predictions=predictions, references=references)
print(f"\n✅ Baseline BLEU Score: {results['score']:.2f}")