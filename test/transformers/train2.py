from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np

torch.cuda.empty_cache()

#dataset = load_dataset('shawhin/imdb-truncated')
dataset = load_dataset("data/uj")
#print(dataset['train'].features)

id2label = {0: "xd", 1: "UJ"}
label2id = {"xd":0, "UJ":1}
#model_id= "openai-community/gpt2"
model_id='distilbert-base-uncased'

model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=2, id2label=id2label, label2id=label2id)

#print(model)

tokenizer=AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    
def tokenizer_fn(x):
    return tokenizer(
        x['text'],
        padding="max_length",
        truncation=True,
        #max_length=128,
    )
    
tokenized_dataset = dataset.map(tokenizer_fn, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=4,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ['q_lin'])
model=get_peft_model(model, peft_config)


training_args = TrainingArguments(
    output_dir="C:/Users/Oliwier/smaug/smaug_test",
    #learning_rate=lr,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator, 
    compute_metrics=compute_metrics,
)

trainer.train()


text_list=["Ile żyje kangur?",
           "UJ jest najlepszy",
           "Dlaczego jestem upośledzony?",
           "Błagam pomocy",
           "Gdzie jest uniwersytet jagielloński?",
           "Czy mogę dostać się na uniwersytet jagielloński?",
           "Czy uniwersytet jagielloński posiada koło studentów informatyki?",
           "Co sądzisz o papieżu?",
           "Jak nazywa się dziekan wydziału informatyki?",
           "Czemu zużywasz tyle ramu?"
           ]
model.to('cpu') # moving to mps for Mac (can alternatively do 'cpu')

print("Trained model predictions:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to("cpu") # moving to mps for Mac (can alternatively do 'cpu')

    logits = model(inputs).logits
    predictions = torch.max(logits,1).indices

    print(text + " - " + id2label[predictions.tolist()[0]])
    
    
from huggingface_hub import login
write_key = 'hf_' 
login(write_key)
hf_name="olipol"
id=hf_name+"/smaug_test"

model.push_to_hub(id)
trainer.push_to_hub(id)