from transformers import pipeline,AutoModel,AutoConfig,AutoTokenizer,DataCollatorWithPadding,AutoModelForSequenceClassification,Trainer,TrainingArguments
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
import evaluate
import datasets
metric = evaluate.load("accuracy")

torch.cuda.empty_cache()

batch_size=4
data = load_dataset("data/uj")
#data=DataLoader(data)
#model_id = "C:/Users/Oliwier/smaug/Llama-3.2-1B"
#model_id = "meta-llama/Llama-3.2-1B"
model_id= "openai-community/gpt2"

#ClassLabels = datasets.ClassLabel(num_classes=2, names=["negative", "positive"],)
data.features=datasets.Features({'text': datasets.Value('string'), 'label': datasets.ClassLabel(num_classes=2, names=["negative", "positive"])})

tokenizer=AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
#classifier = pipeline("text-generation",model=model_id,device_map="auto")
def tokenizer_fn(x):
    return tokenizer(
        #x['sentence1'],
        #x['sentence2'],
        x['text'],
        padding="max_length",
        truncation=True,
        #max_length=128,
    )
    
tokenized_data = data.map(tokenizer_fn)
#print(tokenized_data['train'][0])
#print(data)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model=AutoModelForSequenceClassification.from_pretrained(model_id,num_labels=1)
train_args = TrainingArguments(
    output_dir="C:/Users/Oliwier/smaug/smaug_test",
     per_device_train_batch_size=1,  # Reduce batch size
     per_device_eval_batch_size=1,   # Reduce batch size
     evaluation_strategy="epoch",
    # num_train_epochs=1,
    # logging_dir="C:/Users/Oliwier/smaug/transformers",
    # logging_steps=100,
    # save_steps=100,
    # eval_steps=100,
    # save_total_limit=2,
    # load_best_model_at_end=True,
    # fp16=True,  # Enable mixed precision training
    # gradient_accumulation_steps=2,  # Use gradient accumulation
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



trainer=Trainer(
    model=model,
    args=train_args,
    data_collator=data_collator,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    
)

#predictions = trainer.predict(tokenized_data['validation'])
#print(predictions)
torch.cuda.empty_cache()
trainer.train()
 