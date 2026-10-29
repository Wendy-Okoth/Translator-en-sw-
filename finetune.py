import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# ⚠️ IMPORTANT: Start from Rogendo/en-sw, NOT Helsinki-NLP.
# Your baseline was measured on Rogendo. To compare fairly,
# fine-tune the SAME model — otherwise you're changing two things at once.
model_checkpoint = "Rogendo/en-sw"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
metric = evaluate.load("sacrebleu")

print("Loading dataset...")
raw_datasets = load_dataset(
    "michsethowusu/english-swahili_sentence-pairs",
    split="train[:5000]",
)
raw_datasets = raw_datasets.train_test_split(test_size=0.1, seed=42)

max_length = 128
def preprocess_function(examples):
    inputs = examples["English"]
    targets = examples["Swahili"]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing...")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
args = Seq2SeqTrainingArguments(
    output_dir="./finetuned-en-sw",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,   # 👈 changed from tokenizer=tokenizer (v5 requirement)
)

trainer.train()
trainer.save_model("./finetuned-en-sw-final")
tokenizer.save_pretrained("./finetuned-en-sw-final")
print("✅ Done — model saved to ./finetuned-en-sw-final")