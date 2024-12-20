from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name)
dataset_train = Dataset.from_csv("testXD/data.csv", delimiter=";")
dataset_val = Dataset.from_csv("testXD/data2.csv", delimiter=";")

#print(dataset_train)

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

def preprocess_function(examples):
    inputs = tokenizer(examples["question"], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["answer"], max_length=512, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

print(dataset_train)
#tokenized_datasets = dataset.map(preprocess_function, batched=True)
dataset_train = dataset_train.map(preprocess_function, batched=True)
dataset_val = dataset_val.map(preprocess_function, batched=True)
print(dataset_train)

training_args = TrainingArguments(
    output_dir="./testXD/test",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=data_collator,
)

trainer.train()
    
#model.save_pretrained("./testXD/trained_model")
#tokenizer.save_pretrained("./testXD/trained_model")

