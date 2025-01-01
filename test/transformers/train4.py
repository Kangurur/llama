from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, DatasetDict, load_dataset
import evaluate
import numpy as np
import torch
#import pandas as pd

#dataset = load_dataset("data/uj++")
model_name = "dkleczek/bert-base-polish-cased-v1"  # Polish BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset_train=Dataset.from_csv("data/uj.csv")
dataset_validation=Dataset.from_csv("data/uj2.csv")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True,max_length=512)

#tokenized_dataset = dataset.map(tokenize_function, batched=True)
dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_validation = dataset_validation.map(tokenize_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

class_weights = torch.tensor([1.0, 10.0])  # Dostosowane na podstawie proporcji klas

def compute_loss(model, inputs):
    logits = model(**inputs).logits
    labels = inputs["labels"]
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
    return loss_fn(logits, labels)

training_args = TrainingArguments(
    output_dir="./smaug_part1_v2",           
    evaluation_strategy="epoch",     
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    #logging_dir="./logs",
    #logging_steps=10,
    #save_steps=10,
    #save_total_limit=2,
    save_strategy="epoch",
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    #train_dataset=tokenized_dataset["train"],
    #eval_dataset=tokenized_dataset["validation"],
    train_dataset=dataset_train,
    eval_dataset=dataset_validation,
    tokenizer=tokenizer,
    data_collator=data_collator, 
    #compute_metrics=compute_metrics,
    #compute_loss_func=compute_loss,
)


trainer.train()


#results = trainer.evaluate()
#print("Ewaluacja:", results)

model.to("cuda")

from huggingface_hub import login
write_key = 'hf_' 
login(write_key)
hf_name="olipol"
id=hf_name+"/smaug_part1"

model.push_to_hub(id)
trainer.push_to_hub(id)

def predict(sentence):
    inputs = tokenizer.encode(sentence, return_tensors="pt").to("cuda")
    outputs = model(inputs)
    probs = outputs.logits.softmax(dim=-1)
    return "UJ" if probs[0][1] > probs[0][0] else "nie UJ"

n=int(input())
for i in range(n):
    text = input()
    print(predict(text))