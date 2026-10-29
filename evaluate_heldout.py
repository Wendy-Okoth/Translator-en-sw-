import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# --- Config ---
BASELINE_PATH = "Rogendo/en-sw"
FINETUNED_PATH = "./finetuned-en-sw-final"
EVAL_SLICE = "train[5000:5200]"   # 200 rows NEVER seen during training
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
print(f"Evaluation slice: {EVAL_SLICE} (held out — not in training data)")

# --- Load dataset ONCE (shared by both models) ---
print("\nLoading held-out dataset...")
dataset = load_dataset("michsethowusu/english-swahili_sentence-pairs", split=EVAL_SLICE)
print(f"Loaded {len(dataset)} examples.")

metric = evaluate.load("sacrebleu")

def evaluate_model(name, model_path):
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    predictions = []
    references = []

    for example in tqdm(dataset, desc=name):
        inputs = tokenizer(
            example["English"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            ids = model.generate(**inputs, max_length=512, num_beams=4)

        predictions.append(tokenizer.decode(ids[0], skip_special_tokens=True))
        references.append([example["Swahili"]])

    result = metric.compute(predictions=predictions, references=references)

    # Free memory before loading next model
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return result["score"]

# --- Run both evaluations ---
baseline_bleu = evaluate_model("BASELINE (Rogendo/en-sw)", BASELINE_PATH)
finetuned_bleu = evaluate_model("FINE-TUNED", FINETUNED_PATH)

# --- Final comparison ---
print(f"\n{'='*60}")
print("HELD-OUT EVALUATION RESULTS")
print(f"{'='*60}")
print(f"Slice:                {EVAL_SLICE} (200 unseen examples)")
print(f"Baseline BLEU:        {baseline_bleu:.2f}")
print(f"Fine-tuned BLEU:      {finetuned_bleu:.2f}")
print(f"Improvement:          {finetuned_bleu - baseline_bleu:+.2f} "
      f"({(finetuned_bleu - baseline_bleu) / baseline_bleu * 100:+.1f}%)")
print(f"{'='*60}")