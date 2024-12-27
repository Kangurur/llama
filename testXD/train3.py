from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding

dataset_train = Dataset.from_csv("testXD/data.csv")
dataset_val = Dataset.from_csv("testXD/data2.csv")

model="distilgpt2"
tokenizer=AutoTokenizer.from_pretrained(model)
model=AutoModelForCausalLM.from_pretrained(model)
def tokenize_function(examples):
    return tokenizer(examples["text"])

dataset_train = dataset_train.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
dataset_val = dataset_val.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

#print(dataset_train[0])

training_args = TrainingArguments(
    output_dir="./testXD/test",
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

#do testu